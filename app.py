# v11 FINAL STABLE

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import re, uuid

st.set_page_config(page_title='SpectraKinetics v11.8', layout='wide')

page = st.sidebar.radio("Navigation", ["Spectra Analysis", "Kinetics", "AUC Analysis"])
ex_toggle = st.sidebar.radio("Spectra View", [280, 260])

# =====================
# FILE PARSER
# =====================
def clean_filename(name):
    match = re.search(r"(\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})", name)
    return match.group(1) if match else name[:15]

def parse_file(file_bytes, filename):
    content = file_bytes.decode("utf-8", errors="replace").splitlines()
    spectra, wavelengths, ex_vals, kinetics = {}, [], [], None

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
# FILES
# =====================
files = st.sidebar.file_uploader("Upload", type=["txt"], accept_multiple_files=True)

if files:
    st.session_state.datasets = {}
    for i, f in enumerate(files):
        parsed = parse_file(f.read(), f.name)
        st.session_state.datasets[f"{parsed['filename']}_{i}"] = parsed

data = st.session_state.get("datasets", {})
if not data:
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
            "AUC IF": auc_if
        })

    df = pd.DataFrame(rows)

    st.subheader("Spectra Metrics Table")
    st.dataframe(df, use_container_width=True)

    # ✅ Overlay
    fig = go.Figure()

    for name, d in data.items():
        if ex_toggle in d["spectra"]:
            fig.add_trace(go.Scatter(
                x=d["wavelengths"],
                y=d["spectra"][ex_toggle],
                name=name
            ))

    st.plotly_chart(fig, use_container_width=True, key=f"spectra_{uuid.uuid4()}")

    # ✅ APIES
    st.subheader("APIES (with Regression)")

    fig2 = go.Figure()

    x_vals = df["Index"]

    if len(df) > 1:
        y = df["IR/IF (AUC)"]

        coeffs = np.polyfit(x_vals, y, 1)
        fit = np.polyval(coeffs, x_vals)

        r2 = 1 - np.sum((y-fit)**2)/np.sum((y-np.mean(y))**2)

        fig2.add_trace(go.Scatter(
            x=x_vals, y=y,
            mode="lines+markers",
            name=f"IR/IF (R²={r2:.3f})"
        ))

        fig2.add_trace(go.Scatter(
            x=x_vals, y=fit,
            mode="lines",
            name="IR/IF Fit",
            line=dict(dash="dash")
        ))

    st.plotly_chart(fig2, use_container_width=True)

# =====================
# ✅ AUC ANALYSIS
# =====================
if page == "AUC Analysis":

    st.title("AUC Analysis")

    selected = st.selectbox("Dataset", list(data.keys()))
    d = data[selected]

    if ex_toggle not in d["spectra"]:
        st.warning("Excitation not available")
        st.stop()

    wl = d["wavelengths"]
    y = d["spectra"][ex_toggle]

    min_wl = float(np.min(wl))
    max_wl = float(np.max(wl))

    start_wl = st.number_input("Start WL", min_value=min_wl, max_value=max_wl, value=min_wl+20)
    end_wl = st.number_input("End WL", min_value=min_wl, max_value=max_wl, value=max_wl-20)

    start_wl, end_wl = sorted([start_wl, end_wl])

    mask = (wl >= start_wl) & (wl <= end_wl)
    area = np.trapezoid(y[mask], wl[mask]) if np.any(mask) else 0

    # ✅ Spectrum
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=wl, y=y, name="Spectrum"))

    if np.any(mask):
        fig.add_trace(go.Scatter(
            x=wl[mask],
            y=y[mask],
            fill="tozeroy",
            name="AUC region"
        ))

    st.plotly_chart(fig, use_container_width=True)
    st.metric("AUC", f"{area:.3f}")

    # ✅ Batch AUC WITH LINE GRAPH (FIXED)
    if st.button("Calculate AUC for All Datasets"):

        results = []

        for name, dataset in data.items():

            if ex_toggle not in dataset["spectra"]:
                continue

            wl_f = dataset["wavelengths"]
            y_f = dataset["spectra"][ex_toggle]

            mask = (wl_f >= start_wl) & (wl_f <= end_wl)
            if not np.any(mask):
                continue

            auc_val = np.trapezoid(y_f[mask], wl_f[mask])

            try:
                t = pd.to_datetime(name.split("_")[0])
            except:
                t = name  # fallback

            results.append({"time": t, "AUC": auc_val})

        df_auc = pd.DataFrame(results)

        st.subheader("AUC Table")
        st.dataframe(df_auc)

        # ✅ LINE GRAPH RESTORED
        fig_auc = go.Figure()

        fig_auc.add_trace(go.Scatter(
            x=df_auc["time"],
            y=df_auc["AUC"],
            mode="lines+markers",
            name="AUC vs Time"
        ))

        st.plotly_chart(fig_auc, use_container_width=True)
