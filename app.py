# v12 FINAL STABLE VERIFIED

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import re, uuid

st.set_page_config(page_title='SpectraKinetics v12', layout='wide')

page = st.sidebar.radio("Navigation",
                        ["Spectra Analysis", "Kinetics", "AUC Analysis"])

ex_toggle = st.sidebar.radio("Spectra View", [280, 260])

# =====================
# PARSER
# =====================
def clean_filename(name):
    match = re.search(r"(\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})", name)
    return match.group(1) if match else name[:15]

def parse_file(file_bytes, filename):
    content = file_bytes.decode("utf-8", errors="replace").splitlines()

    spectra, wavelengths, ex_vals = {}, [], []

    for i, line in enumerate(content):
        if "excitation wavelength" in line.lower():
            parts = line.split("\t")
            ex_vals = [float(v) for v in parts[2:] if v]
        if line.split("\t")[0].isdigit():
            start = i
            break

    matrix = []

    for line in content[start:]:
        parts = line.split("\t")
        try:
            wavelengths.append(float(parts[1]))
            matrix.append([float(x) for x in parts[2:2+len(ex_vals)]])
        except:
            continue

    matrix = np.array(matrix) if len(matrix) > 0 else np.array([])
    for j, ex in enumerate(ex_vals if len(matrix) > 0 else []):
        spectra[ex] = matrix[:, j]

    return {
        "wavelengths": np.array(wavelengths),
        "spectra": spectra,
        "filename": clean_filename(filename),
        "kinetics": None
    }

# =====================
# LOAD FILES
# =====================
files = st.sidebar.file_uploader("Upload", type=["txt"], accept_multiple_files=True)

if files:
    st.session_state.datasets = {}
    for i, f in enumerate(files):
        parsed = parse_file(f.read(), f.name)
        st.session_state.datasets[f"{parsed['filename']}_{i}"] = parsed

data = st.session_state.get("datasets", {})
if not data:
    st.info("Upload data to begin.")
    st.stop()

# =====================
# ✅ SPECTRA ANALYSIS
# =====================
if page == "Spectra Analysis":

    st.title("Spectra Analysis")

    col1, col2 = st.columns(2)

    with col1:
        ir_start = st.number_input("IR Start", value=270.0)
        ir_end = st.number_input("IR End", value=300.0)

    with col2:
        if_start = st.number_input("IF Start", value=320.0)
        if_end = st.number_input("IF End", value=390.0)

    ir_start, ir_end = sorted([ir_start, ir_end])
    if_start, if_end = sorted([if_start, if_end])

    rows = []

    for i, (name, d) in enumerate(data.items()):

        if ex_toggle not in d["spectra"]:
            continue

        wl = d["wavelengths"]
        y = d["spectra"][ex_toggle]

        mask_ir = (wl >= ir_start) & (wl <= ir_end)
        mask_if = (wl >= if_start) & (wl <= if_end)

        auc_ir = np.trapezoid(y[mask_ir], wl[mask_ir]) if np.any(mask_ir) else np.nan
        auc_if = np.trapezoid(y[mask_if], wl[mask_if]) if np.any(mask_if) else np.nan

        irif = auc_ir / auc_if if (not np.isnan(auc_if) and auc_if != 0) else np.nan

        rows.append({
            "File": name,
            "Index": i,
            "IR/IF (AUC)": irif,
            "AUC IR": auc_ir,
