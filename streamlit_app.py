import streamlit as st
import yaml
import subprocess
import shutil
import requests
import os
import logging
import sys
import urllib.parse
import tarfile
import re
from pathlib import Path
from google import genai

# Page Config
st.set_page_config(
    page_title="SudoDocs OAS Validator",
    page_icon="📘",
    layout="wide"
)

# --- CRITICAL FIX: Manually Install Node.js v20 ---
def ensure_node_installed():
    node_version = "v20.11.0"
    install_dir = Path("./node_runtime")
    
    node_dirname = f"node-{node_version}-linux-x64"
    node_bin_path = install_dir / node_dirname / "bin"
    
    try:
        result = subprocess.run(["node", "-v"], capture_output=True, text=True)
        if result.returncode == 0 and result.stdout.strip().startswith("v20"):
            return 
    except FileNotFoundError:
        pass

    if not node_bin_path.exists():
        with st.spinner(f"🔧 Installing Node.js {node_version} (Required for ReadMe v10)..."):
            url = f"https://nodejs.org/dist/{node_version}/{node_dirname}.tar.xz"
            response = requests.get(url, stream=True)
            if response.status_code == 200:
                tar_path = Path("node.tar.xz")
                with open(tar_path, 'wb') as f: f.write(response.raw.read())
                with tarfile.open(tar_path) as tar: tar.extractall(install_dir)
                os.remove(tar_path)
            else:
                st.error("Failed to download Node.js runtime."); st.stop()
    
    try:
        extracted_folder = list(install_dir.glob("node-v*-linux-x64"))[0]
        bin_path = extracted_folder / "bin"
        os.environ["PATH"] = f"{str(bin_path.absolute())}{os.pathsep}{os.environ['PATH']}"
    except IndexError:
        st.error("Node.js installation failed: Extracted folder not found."); st.stop()

    # --- UPSELL BANNER START ---
    with st.sidebar:
        st.info("🚀 **Docs valid, but lying?**")
        st.markdown(
            """
            This tool fixes **syntax** errors, but it can't tell if your API spec actually matches your code.
            
            **SudoDocs** unit-tests your documentation against your Git diffs to catch **Logic Drift**.
            
            [👉 **Try the Drift Inspector**](https://sudodocs.com?utm_source=oas_validator_sidebar)
            """
        )
        st.divider()
    # --- UPSELL BANNER END ---

# --- Initialize Session State for Logs ---
if 'logs' not in st.session_state: st.session_state.logs = []

class StreamlitLogHandler(logging.Handler):
    def __init__(self, container, download_placeholder=None):
        super().__init__()
        self.container = container
        self.download_placeholder = download_placeholder
    def emit(self, record):
        msg = self.format(record)
        st.session_state.logs.append(msg)
        self.container.code("\n".join(st.session_state.logs), language="text")

# --- Helper Functions ---
def get_npx_path(): return shutil.which("npx")

def validate_env(api_key, required=True):
    if not api_key and required:
        st.error("❌ ReadMe API Key is missing. Please enter it in the sidebar."); st.stop()
    return True

# UPDATED: Added cwd argument to run commands in the workspace directory
def run_command(command_list, log_logger, cwd=None):
    try:
        cmd_str = " ".join(command_list)
        dir_msg = f" (in {cwd})" if cwd else ""
        log_logger.info(f"Running: {cmd_str}{dir_msg}")
        
        process = subprocess.Popen(
            command_list, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding='utf-8', env=os.environ.copy(), cwd=cwd
        )
        for line in process.stdout:
            if line.strip(): log_logger.info(f"[CLI] {line.strip()}")
        process.wait()
        return process.returncode
    except Exception as e:
        log_logger.error(f"❌ Command failed: {e}"); return 1

# --- AI Logic ---
def analyze_errors_with_ai(log_content, api_key, model_name):
    if not api_key: return None
    try:
        client = genai.Client(api_key=api_key)
        prompt = f"Analyze logs:\n{log_content}"
        response = client.models.generate_content(model=model_name, contents=[prompt])
        return response.text
    except Exception as e: return f"AI Error: {e}"

# --- NEW: Smart ID Lookup (Requested Feature) ---
def get_api_id_smart(api_title, api_key, logger):
    """Finds ID even if words are reordered (Fuzzy Match)."""
    headers = {"Authorization": f"Basic {api_key}", "Accept": "application/json"}
    base_url = "https://dash.readme.com/api/v1"
    
    try:
        logger.info(f"🔎 Looking for ID for: '{api_title}'")
        def tokenize(text): return set(re.findall(r'\w+', text.lower()))
        target_tokens = tokenize(api_title)
        
        # Fetch all APIs
        res = requests.get(f"{base_url}/api-specification", headers=headers, params={"perPage": 100})
        
        if res.status_code == 200:
            apis = res.json()
            # 1. Exact Match
            for api in apis:
                if api["title"] == api_title: return api["_id"], api["title"]
            # 2. Fuzzy Match
            for api in apis:
                if target_tokens == tokenize(api["title"]):
                    logger.info(f"✨ Smart Match found: '{api['title']}'")
                    return api["_id"], api["title"]
            logger.warning(f"⚠️ No matching API found for '{api_title}'")
        else:
            logger.error(f"❌ ReadMe API Error: {res.status_code}")
    except Exception as e:
        logger.error(f"❌ ID Lookup Exception: {e}")
    return None, None

# --- Git Logic ---
def setup_git_repo(repo_url, repo_dir, git_token, git_username, branch_name, logger):
    logger.info(f"🚀 Starting Git Operation for branch: {branch_name}...")
    repo_path = Path(repo_dir)
    
    if repo_url: repo_url = repo_url.strip().strip('"').strip("'")
    if repo_url and repo_url.count("https://") > 1:
        match = re.search(r"(https://github\.com/.*)$", repo_url)
        if match: repo_url = match.group(1)

    auth_repo_url = repo_url
    if repo_url and git_username and git_token:
        parsed = urllib.parse.urlparse(repo_url)
        clean_netloc = parsed.netloc.split("@")[-1]
        auth_repo_url = urllib.parse.urlunparse((parsed.scheme, f"{git_username}:{git_token}@{clean_netloc}", parsed.path, parsed.params, parsed.query, parsed.fragment))

    env_vars = os.environ.copy(); env_vars["GIT_TERMINAL_PROMPT"] = "0"
    
    if not repo_path.exists():
        if run_command(["git", "clone", "--depth", "1", "--branch", branch_name, auth_repo_url, str(repo_path)], logger) != 0:
            st.error("Git Clone Failed."); st.stop()
    else:
        try:
            subprocess.run(["git", "-C", str(repo_path), "remote", "set-url", "origin", auth_repo_url], check=True, env=env_vars)
            subprocess.run(["git", "-C", str(repo_path), "fetch", "origin", branch_name], check=True, env=env_vars)
            subprocess.run(["git", "-C", str(repo_path), "reset", "--hard", f"origin/{branch_name}"], check=True, env=env_vars)
        except: logger.warning("Git update failed, using existing files.")

def delete_repo(repo_dir):
    path = Path(repo_dir)
    if path.exists(): shutil.rmtree(path); return True, "Deleted successfully."
    return False, "Path does not exist."

# --- File Operations (FIXED: Nesting Logic) ---
def prepare_files(filename, paths, workspace, dependency_list, logger):
    source = None
    main_candidate = Path(paths['specs']) / f"{filename}.yaml"
    if main_candidate.exists(): source = main_candidate
    elif paths.get('secondary') and (Path(paths['secondary']) / f"{filename}.yaml").exists():
        source = Path(paths['secondary']) / f"{filename}.yaml"

    if not source: logger.error(f"❌ Source '{filename}.yaml' not found."); st.stop()

    workspace_path = Path(workspace)
    workspace_path.mkdir(parents=True, exist_ok=True)
    
    # --- FIX: Preserve directory structure ---
    try:
        # Calculate where the file is relative to the "specs" root
        rel_path = source.resolve().relative_to(paths['specs'].resolve())
        destination = workspace_path / rel_path
        destination.parent.mkdir(parents=True, exist_ok=True)
    except ValueError:
        destination = workspace_path / source.name

    shutil.copy(source, destination)
    logger.info(f"📂 Copied YAML to workspace: {destination.relative_to(workspace_path)}")

    for folder in dependency_list:
        clean = folder.strip()
        if not clean: continue
        src = Path(paths['specs']) / clean
        dest = workspace_path / clean
        if src.exists():
            if dest.exists(): shutil.rmtree(dest)
            shutil.copytree(src, dest)
            logger.info(f"📂 Copied dependency: {clean}")
    return destination

def process_yaml_content(file_path, api_domain, logger):
    logger.info("🛠️ Injecting x-readme extensions...")
    try:
        with open(file_path, "r") as f: data = yaml.safe_load(f)

        if "openapi" in data:
            pos = list(data.keys()).index("openapi")
            items = list(data.items())
            items.insert(pos + 1, ("x-readme", {"explorer-enabled": False}))
            data = dict(items)
        
        domain = api_domain if api_domain else "example.com"
        if "servers" not in data or not data["servers"]: data["servers"] = [{"url": f"https://{domain}", "variables": {}}]
        if "variables" not in data["servers"][0]: data["servers"][0]["variables"] = {}
        data["servers"][0]["variables"]["base-url"] = {"default": domain}
        data["servers"][0]["variables"]["protocol"] = {"default": "https"}

        # Save edited file in SAME directory
        edited_path = file_path.parent / (file_path.stem + "_edited.yaml")
        with open(edited_path, "w") as f: yaml.dump(data, f, sort_keys=False)
        return edited_path
    except Exception as e: logger.error(f"❌ YAML Error: {e}"); st.stop()

# --- CALLBACKS ---
def clear_credentials():
    st.session_state.readme_key = ""
    st.session_state.git_user = ""
    st.session_state.git_token = ""
    st.session_state.logs = []

def clear_logs():
    st.session_state.logs = []

# --- MAIN ---
def main():
    ensure_node_installed()

    # --- UPSELL BANNER START ---
    with st.sidebar:
        st.info("🚀 **Docs valid, but lying?**")
        st.markdown(
            """
            This tool fixes **syntax** errors, but it can't tell if your API spec actually matches your code.
            
            **SudoDocs** unit-tests your documentation against your Git diffs to catch **Logic Drift**.
            
            [👉 **Try the Drift Inspector**](https://sudodocs.com?utm_source=oas_validator_sidebar)
            """
        )
        st.divider()
    # --- UPSELL BANNER END ---

    st.sidebar.title("⚙️ Refactored Config")
    if 'readme_key' not in st.session_state: st.session_state.readme_key = ""
    if 'gemini_key' not in st.session_state: st.session_state.gemini_key = ""
    
    readme_key = st.sidebar.text_input("ReadMe API Key", key="readme_key", type="password")
    gemini_key = st.sidebar.text_input("Gemini API Key", key="gemini_key", type="password")
    
    st.sidebar.subheader("Git Repo Config")
    repo_path = st.sidebar.text_input("Local Clone Path", value="./cloned_repo")
    if st.sidebar.button("🗑️ Reset Repo"):
        if delete_repo(repo_path): st.sidebar.success("Deleted successfully.")

    repo_url = st.sidebar.text_input("Git Repo URL", key="repo_url")
    branch_name = st.sidebar.text_input("Branch Source", value="main")
    git_user = st.sidebar.text_input("Git User", key="git_user")
    git_token = st.sidebar.text_input("Git Token", key="git_token", type="password")
    st.sidebar.button("🔒 Clear Credentials", on_click=clear_credentials)

    st.sidebar.subheader("Internal Paths")
    spec_rel_path = st.sidebar.text_input("Main Specs Path", value="specs")
    secondary_rel_path = st.sidebar.text_input("Secondary Path", value="")
    dep_input = st.sidebar.text_input("Dependencies", value="common")
    dependency_list = [x.strip() for x in dep_input.split(",")]
    api_domain = st.sidebar.text_input("API Domain", value="api.example.com")

    abs_spec_path = Path(repo_path) / spec_rel_path
    paths = {"repo": repo_path, "specs": abs_spec_path}
    if secondary_rel_path: paths["secondary"] = Path(repo_path) / secondary_rel_path
    workspace_dir = "./temp_workspace"

    st.title("🚀 SudoDocs OAS Validator")

    col1, col2 = st.columns(2)
    with col1:
        files = []
        if abs_spec_path.exists(): files.extend([f.stem for f in abs_spec_path.glob("*.yaml")])
        if "secondary" in paths and paths["secondary"].exists(): files.extend([f.stem for f in paths["secondary"].glob("*.yaml")])
        files = sorted(list(set(files)))
        selected_file = st.selectbox("Select File", files) if files else st.text_input("Filename", "audit")
    with col2: target_branch = st.text_input("Target ReadMe Branch", "main")

    st.markdown("### Settings")
    c1, c2, c3 = st.columns(3)
    use_swagger = c1.checkbox("Swagger", True)
    use_redocly = c2.checkbox("Redocly", True)
    use_readme = c3.checkbox("ReadMe CLI", False)

    c_btn1, c_btn2 = st.columns(2)
    btn_validate = c_btn1.button("🔍 Validate")
    btn_upload = c_btn2.button("🚀 Upload", type="primary")

    log_container = st.empty()
    download_placeholder = st.empty()

    if btn_validate or btn_upload:
        st.session_state.logs = []
        logger = logging.getLogger("streamlit_logger")
        logger.setLevel(logging.INFO)
        if logger.handlers: logger.handlers = []
        handler = StreamlitLogHandler(log_container, download_placeholder)
        logger.addHandler(handler)

        has_key = validate_env(readme_key, required=btn_upload)
        npx_path = get_npx_path()
        if not npx_path: logger.error("❌ NodeJS/npx not found."); st.stop()

        setup_git_repo(repo_url, repo_path, git_token, git_user, branch_name, logger)
        final_yaml_path = prepare_files(selected_file, paths, workspace_dir, dependency_list, logger)
        edited_file = process_yaml_content(final_yaml_path, api_domain, logger)
        
        # --- CRITICAL FIX FOR PATHS ---
        # Instead of running from workspace_dir, we run from the PARENT FOLDER of the edited file.
        # This aligns CWD with the file location, so '../common' works correctly.
        abs_execution_dir = edited_file.parent.resolve()
        target_filename = edited_file.name

        failed = False
        if use_swagger and run_command([npx_path, "--yes", "swagger-cli", "validate", target_filename], logger, cwd=abs_execution_dir) != 0: failed = True
        if use_redocly and run_command([npx_path, "--yes", "@redocly/cli@1.25.0", "lint", target_filename], logger, cwd=abs_execution_dir) != 0: failed = True
        if use_readme and has_key and run_command([npx_path, "--yes", "rdme", "openapi:validate", target_filename], logger, cwd=abs_execution_dir) != 0: failed = True

        if failed:
            st.error("Validation Failed.")
            if btn_upload: st.stop()
        elif btn_upload:
            logger.info("🚀 Preparing Upload...")
            
            with open(edited_file, "r") as f:
                ydata = yaml.safe_load(f)
                ytitle = ydata.get("info", {}).get("title", "")
            
            # Smart Match Logic
            api_id, matched_title = get_api_id_smart(ytitle, readme_key, logger)
            
            if api_id and matched_title and matched_title != ytitle:
                logger.info(f"🔧 Auto-correcting title: '{ytitle}' -> '{matched_title}'")
                ydata["info"]["title"] = matched_title
                with open(edited_file, "w") as f: yaml.dump(ydata, f, sort_keys=False)

            cmd = [npx_path, "--yes", "rdme", "openapi", "upload", target_filename, "--key", readme_key, "--branch", target_branch]
            
            if api_id:
                cmd.extend(["--id", api_id])
            else:
                logger.warning("⚠️ No matching ID found. 'rdme' will attempt to create a NEW API.")

            if run_command(cmd, logger, cwd=abs_execution_dir) == 0:
                st.success("✅ Uploaded successfully!")
            else:
                st.error("❌ Upload failed.")
        else:
            st.success("Validation Passed.")

    with st.expander("Downloads & Tools"):
        if 'last_edited_file' in st.session_state: # Fixed check
             path = Path(st.session_state.last_edited_file) # Fixed access
             if path.exists():
                 with open(path, "r") as f: st.download_button("Download YAML", f.read(), path.name)
        if st.session_state.logs: st.button("Clear Logs", on_click=clear_logs)

    if st.session_state.logs and gemini_key:
        if st.button("Analyze Logs with AI"):
             with st.spinner("Analyzing..."):
                analysis = analyze_errors_with_ai("\n".join(st.session_state.logs), gemini_key, "gemini-2.0-flash")
                st.markdown(analysis)

if __name__ == "__main__": main()
