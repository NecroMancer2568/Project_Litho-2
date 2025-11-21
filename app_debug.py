import streamlit as st
import os
import joblib
import traceback

st.set_page_config(page_title="Debug: repo files & model loader", layout="wide")
st.title("Debug: repository files & model loader")

st.subheader("Working directory and top-level listing")
st.write("CWD:", os.getcwd())
st.write("Top-level files/folders:")
st.write(sorted(os.listdir(".")))

st.subheader("Full repo walk (showing folders and counts)")
for root, dirs, files in os.walk("."):
    # limit output size a bit by skipping common heavy dirs if present
    if any(skip in root for skip in [".git", "__pycache__", ".streamlit"]):
        continue
    st.write(f"{root} — dirs: {len(dirs)}, files: {len(files)}")
    # if there are model-like names in this folder, print them
    model_like = [f for f in files if any(tok in f.lower() for tok in ['model', 'model.pkl', '.pkl', '.joblib', 'korba'])]
    if model_like:
        st.write("  model-like files:", model_like)

st.subheader("Search for 'korba' and common model filenames")
candidates = []
for root, dirs, files in os.walk("."):
    for f in files:
        if 'korba' in f.lower() or 'korba_mine' in f.lower() or 'model' in f.lower() or f.lower().endswith(('.pkl', '.joblib', '.sav')):
            candidates.append(os.path.join(root, f))
if not candidates:
    st.error("No candidate model files found in the runtime filesystem.")
else:
    st.success(f"Found {len(candidates)} candidate file(s):")
    for c in candidates:
        try:
            st.write(c, " — size:", os.path.getsize(c), "bytes")
        except Exception:
            st.write(c, " — size: n/a")

st.subheader("Attempt to load candidate model files (safe attempt)")
for c in candidates:
    st.write("Trying to load:", c)
    try:
        m = joblib.load(c)
        st.success(f"Loaded OK: {c} (type: {type(m)})")
    except Exception as e:
        st.error(f"Failed to load: {c}")
        st.text(traceback.format_exc())