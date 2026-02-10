import streamlit as st
import streamlit.components.v1 as components

# DESTINATION: Your NEW SudoDocs App
NEW_URL = "https://sudodocs-oas-validator.streamlit.app/"

st.set_page_config(page_title="Redirecting...", layout="wide")

# JavaScript to move them instantly
components.html(
    f"""
    <script>
        window.location.href = "{NEW_URL}";
    </script>
    """,
    height=0,
)

st.title("Redirecting to SudoDocs... 🚀")
st.markdown(f"If you are not redirected, [click here]({NEW_URL}).")
