import streamlit as st
import yaml
import subprocess
import shutil
import requests
import os
import logging
import re
import urllib.parse
from pathlib import Path
from google import genai

# Page Config
st.set_page_config(page_title="ReadMe Validator", layout="wide")

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

# --- Helpers ---
def get_npx_path(): return shutil.which("npx")

def validate_env(api_key, required=True):
    if not api_key and required:
        st.error("❌ ReadMe API Key is missing."); st.stop()
    return bool(api_key)

def run_command(command_list, log_logger, cwd=None):
    try:
        cmd_str = " ".join(command_list)
        dir_msg = f" (in {cwd})" if cwd else ""
        log_logger.info(f"Running: {cmd_str}{dir_msg}")
        process = subprocess.Popen(command_list, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', cwd=cwd)
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
        response = client.models.generate_content(model=model_name, contents=[f"Analyze logs:\n{log_content}"])
        return response.text
    except Exception as e: return f"AI Error: {e}"

def apply_ai_fixes(original_path, log_content, api_key, model_name):
    if not api_key: return None
    try:
        with open(original_path, 'r') as f: content = f.read()
        client = genai.Client(api_key=api_key)
        prompt = f"Fix YAML errors. Preserve x-readme/servers/info. Return ONLY YAML.\nLogs: {log_content}\nYAML: {content}"
        response = client.models.generate_content(model=model_name, contents=[prompt])
        match = re.search(r'```yaml\n(.*?)\n```', response.text, re.DOTALL)
        return match.group(1) if match else response.text
    except: return None

# --- Git Logic ---
def setup_git_repo(repo_url, repo_dir, git_token, git_username, branch_name, logger):
    logger.info(f"🚀 Git: Branch '{branch_name}'...")
    repo_path = Path(repo_dir)
    
    # Auth URL Construction
    if repo_url.count("https://") > 1: repo_url = re.search(r"(https://github\.com/.*)$", repo_url).group(1)
    parsed = urllib.parse.urlparse(repo_url)
    safe_user = urllib.parse.quote(git_username.strip(), safe='')
    safe_token = urllib.parse.quote(git_token.strip(), safe='')
    auth_url = urllib.parse.urlunparse((parsed.scheme, f"{safe_user}:{safe_token}@{parsed.netloc.split('@')[-1]}", parsed.path, parsed.params, parsed.query, parsed.fragment))

    clean_env = os.environ.copy(); clean_env["GIT_TERMINAL_PROMPT"] = "0"
    
    if not repo_path.exists():
        try:
            subprocess.run(["git", "clone", "--depth", "1", "--branch", branch_name, auth_url, str(repo_path)], check=True, capture_output=True, env=clean_env)
            logger.info("✅ Repo cloned.")
        except subprocess.CalledProcessError as e: logger.error(f"❌ Clone Failed: {e.stderr}"); st.stop()
    else:
        try:
            subprocess.run(["git", "-C", str(repo_path), "remote", "set-url", "origin", auth_url], check=True, capture_output=True, env=clean_env)
            subprocess.run(["git", "-C", str(repo_path), "fetch", "origin", branch_name], check=True, capture_output=True, env=clean_env)
            subprocess.run(["git", "-C", str(repo_path), "checkout", branch_name], check=True, capture_output=True, env=clean_env)
            subprocess.run(["git", "-C", str(repo_path), "pull", "origin", branch_name], check=True, capture_output=True, env=clean_env)
            logger.info(f"✅ Switched to '{branch_name}'.")
        except subprocess.CalledProcessError as e: logger.error(f"❌ Git Update Failed: {e}")

def delete_repo(repo_dir):
    if Path(repo_dir).exists(): shutil.rmtree(repo_dir); return True, "Deleted."
    return False, "Not found."

# --- File Ops (NEW: COPY ALL STRATEGY) ---
def prepare_files(filename, paths, workspace, logger):
    # 1. Clean Workspace
    ws_path = Path(workspace)
    if ws_path.exists(): shutil.rmtree(ws_path)
    
    # 2. Copy the ENTIRE Main Specs folder structure
    # This preserves 'common', 'logical_metadata', etc. exactly as they are in the repo.
    specs_src = paths['specs']
    shutil.copytree(specs_src, ws_path)
    logger.info(f"📂 Copied entire specs folder to workspace.")

    # 3. Find the selected file inside the new workspace
    # We look for the file recursively because we don't know if it's in root or subfolder
    found_files = list(ws_path.rglob(f"{filename}.yaml"))
    
    if not found_files:
        logger.error(f"❌ Could not find '{filename}.yaml' inside the specs folder.")
        st.stop()
    
    # If multiple files with same name (rare), pick the one that matches the nesting
    # For now, pick the first exact match
    target_file = found_files[0]
    logger.info(f"✅ Located target file: {target_file.relative_to(ws_path)}")
    return target_file

def process_yaml_content(file_path, version, api_domain, logger):
    logger.info("🛠️ Injecting extensions...")
    try:
        with open(file_path, "r") as f: data = yaml.safe_load(f)
        
        # Inject x-readme
        if "openapi" in data:
            pos = list(data.keys()).index("openapi")
            items = list(data.items())
            items.insert(pos + 1, ("x-readme", {"explorer-enabled": False}))
            data = dict(items)
        
        # Inject Servers
        data["info"]["version"] = version
        domain = api_domain if api_domain else "example.com"
        if "servers" not in data or not data["servers"]: data["servers"] = [{"url": f"https://{domain}", "variables": {}}]
        if "variables" not in data["servers"][0]: data["servers"][0]["variables"] = {}
        data["servers"][0]["variables"]["base-url"] = {"default": domain}
        data["servers"][0]["variables"]["protocol"] = {"default": "https"}

        edited_path = file_path.parent / (file_path.stem + "_edited.yaml")
        with open(edited_path, "w") as f: yaml.dump(data, f, sort_keys=False)
        return edited_path
    except Exception as e: logger.error(f"❌ YAML Error: {e}"); st.stop()

# --- ReadMe Helpers ---
def check_and_create_version(version, api_key, base_url, logger, create_if_missing):
    if not api_key: return
    headers = {"Authorization": f"Basic {api_key}", "Accept": "application/json"}
    try:
        res = requests.get(f"{base_url}/version", headers=headers)
        if res.status_code == 200 and any(v["version"] == version for v in res.json()):
            logger.info(f"✅ Version '{version}' exists."); return
        if create_if_missing:
            logger.info(f"⚠️ Creating version '{version}'...")
            fork = res.json()[0]['version'] if res.json() else "latest"
            requests.post(f"{base_url}/version", headers=headers, json={"version": version, "is_stable": False, "from": fork})
    except Exception as e: logger.error(f"❌ Version check failed: {e}")

def get_api_id(api_name, version, api_key, base_url, logger):
    if not api_key: return None, None
    headers = {"Authorization": f"Basic {api_key}", "Accept": "application/json", "x-readme-version": version}
    try:
        def tokenize(text): return set(re.findall(r'\w+', text.lower()))
        target_tokens = tokenize(api_name)
        res = requests.get(f"{base_url}/api-specification", headers=headers, params={"perPage": 100})
        if res.status_code == 200:
            for api in res.json():
                if api["title"] == api_name: return api["_id"], api["title"]
            for api in res.json():
                if target_tokens == tokenize(api["title"]): return api["_id"], api["title"]
    except Exception as e: logger.error(f"❌ ID Lookup Error: {e}")
    return None, None

def create_new_api_via_requests(file_path, version, api_key, base_url, logger):
    logger.info("📤 Creating NEW API via direct upload...")
    headers = {"Authorization": f"Basic {api_key}", "x-readme-version": version}
    try:
        with open(file_path, 'rb') as f:
            res = requests.post(f"{base_url}/api-specification", headers=headers, files={'spec': (file_path.name, f)})
        if res.status_code in [200, 201]: 
            new_id = res.json().get("_id")
            logger.info(f"✅ Created! ID: {new_id}"); return new_id
        else: logger.error(f"❌ Failed: {res.text}"); return None
    except Exception as e: logger.error(f"❌ Exception: {e}"); return None

# --- Main ---
def main():
    st.sidebar.title("⚙️ Config")
    for k in ['readme_key', 'gemini_key', 'git_user', 'git_token', 'repo_url', 'last_edited_file', 'corrected_file']:
        if k not in st.session_state: st.session_state[k] = "" if 'file' not in k else None
    
    readme_key = st.sidebar.text_input("ReadMe Key", key="readme_key", type="password")
    gemini_key = st.sidebar.text_input("Gemini Key", key="gemini_key", type="password")
    
    st.sidebar.subheader("Git")
    repo_path = st.sidebar.text_input("Clone Path", value="./cloned_repo")
    if st.sidebar.button("🗑️ Delete Repo"): delete_repo(repo_path)
    repo_url = st.sidebar.text_input("Repo URL", key="repo_url")
    branch = st.sidebar.text_input("Branch", value="main")
    git_user = st.sidebar.text_input("User", key="git_user")
    git_token = st.sidebar.text_input("Token", key="git_token", type="password")
    st.sidebar.button("🔒 Clear Creds", on_click=lambda: st.session_state.update(readme_key="", git_token="", logs=[]))

    st.sidebar.subheader("Paths")
    spec_rel = st.sidebar.text_input("Main Specs Path", value="specs")
    sec_rel = st.sidebar.text_input("Secondary Path (Opt)", value="")
    # Not strictly needed with new logic, but kept for UI consistency
    st.sidebar.text_input("Dependency Folders", value="common")
    domain = st.sidebar.text_input("API Domain", value="api.example.com")

    abs_spec = Path(repo_path) / spec_rel
    paths = {"repo": repo_path, "specs": abs_spec}
    if sec_rel: paths["secondary"] = Path(repo_path) / sec_rel
    workspace = "./temp_workspace"

    st.title("🚀 OpenAPI Validator")
    
    c1, c2 = st.columns(2)
    with c1:
        files = []
        if abs_spec.exists(): files.extend([f.stem for f in abs_spec.glob("*.yaml")])
        if "secondary" in paths and paths["secondary"].exists(): files.extend([f.stem for f in paths["secondary"].glob("*.yaml")])
        files = sorted(list(set(files)))
        sel_file = st.selectbox("File", files) if files else st.text_input("Filename", "audit")
    with c2: version = st.text_input("Version", "1.0")
    
    st.markdown("---")
    u_opts = ["Original (Edited)"]; 
    if st.session_state.corrected_file: u_opts.append("AI Corrected")
    u_choice = st.radio("Upload:", u_opts, horizontal=True)
    
    col1, col2 = st.columns(2)
    b_val = col1.button("🔍 Validate")
    b_up = col2.button(f"🚀 Upload: {u_choice}", type="primary")

    log_con = st.empty()
    if st.session_state.logs: log_con.code("\n".join(st.session_state.logs))

    if b_val or b_up:
        st.session_state.logs = []
        logger = logging.getLogger("st"); logger.handlers = []
        logger.addHandler(StreamlitLogHandler(log_con))
        logger.setLevel(logging.INFO)

        has_key = validate_env(readme_key, required=bool(b_up))
        npx = get_npx_path()
        base_url = "https://dash.readme.com/api/v1"

        setup_git_repo(repo_url, repo_path, git_token, git_user, branch, logger)
        
        # --- NEW PREPARE LOGIC ---
        logger.info("📂 Setting up workspace...")
        target_file = prepare_files(sel_file, paths, workspace, logger)
        abs_ws = Path(workspace).resolve()
        
        # Calculate relative path for CLI commands
        rel_target = target_file.relative_to(abs_ws)

        if has_key: check_and_create_version(version, readme_key, base_url, logger, bool(b_up))
        
        edited = process_yaml_content(target_file, version, domain, logger)
        # Update path to point to the edited version
        final_target = edited
        rel_final_target = final_target.relative_to(abs_ws)
        
        if b_up and u_choice == "AI Corrected" and st.session_state.corrected_file:
            final_target = Path(st.session_state.corrected_file).resolve()
            rel_final_target = final_target.relative_to(abs_ws)

        fail = False
        if b_up or st.checkbox("Swagger", True):
            if run_command([npx, "--yes", "swagger-cli", "validate", str(rel_final_target)], logger, cwd=abs_ws) != 0: fail = True
        
        if not b_up and st.checkbox("Redocly", True):
             if run_command([npx, "--yes", "@redocly/cli@1.25.0", "lint", str(rel_final_target)], logger, cwd=abs_ws) != 0: fail = True

        if (b_up or st.checkbox("ReadMe", False)) and has_key:
             if run_command([npx, "--yes", "rdme@8", "openapi:validate", str(rel_final_target)], logger, cwd=abs_ws) != 0: fail = True

        if fail: st.error("Errors found."); logger.error("❌ Validation Failed.")
        elif b_up:
            logger.info("🚀 Uploading...")
            with open(final_target, "r") as f: 
                yd = yaml.safe_load(f); yt = yd.get("info", {}).get("title", "")
            
            aid, mtitle = get_api_id(yt, version, readme_key, base_url, logger)
            
            if aid and mtitle != yt:
                logger.info(f"🔧 Renaming to: {mtitle}")
                yd["info"]["title"] = mtitle; 
                with open(final_target, "w") as f: yaml.dump(yd, f, sort_keys=False)

            if aid:
                cmd = [npx, "--yes", "rdme@8", "openapi", str(rel_final_target), "--useSpecVersion", "--version", version, "--id", aid, "--key", readme_key]
                if run_command(cmd, logger, cwd=abs_ws) == 0: st.success("Success!")
            else:
                logger.warning("⚠️ New API detected. Bundling...")
                bun = f"{final_target.stem}_bundled.yaml"
                if run_command([npx, "--yes", "swagger-cli", "bundle", str(rel_final_target), "-o", bun, "-t", "yaml"], logger, cwd=abs_ws) == 0:
                    if create_new_api_via_requests(abs_ws / bun, version, readme_key, base_url, logger): st.success("Created!")
        else: st.success("Valid.")
    
    if st.session_state.logs and st.button("🗑️ Clear"): st.session_state.logs = []
    
    # AI Fix Logic
    if st.session_state.logs and gemini_key:
        c1, c2 = st.columns(2)
        if c1.button("🧐 Analyze"): 
            st.markdown(analyze_errors_with_ai("\n".join(st.session_state.logs), gemini_key, "gemini-2.5-pro"))
        if c2.button("✨ Auto-Fix"):
            fix = apply_ai_fixes(st.session_state.last_edited_file or target_file, "\n".join(st.session_state.logs), gemini_key, "gemini-2.5-pro")
            if fix:
                p = Path(st.session_state.last_edited_file or target_file)
                cp = p.parent / (p.stem.replace("_edited", "") + "_corrected.yaml")
                with open(cp, "w") as f: f.write(fix)
                st.session_state.corrected_file = str(cp)
                st.success("Fixed!")

if __name__ == "__main__": main()
