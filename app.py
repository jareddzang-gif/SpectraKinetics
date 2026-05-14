# v14 FINAL (FULLY FIXED + APIES DASHBOARD)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import re

st.set_page_config(page_title='SpectraKinetics v14', layout='wide')

page = st.sidebar.radio("Navigation",
                        ["APIES Dashboard", "Kinetics", "AUC Analysis"])

ex_toggle = st.sidebar.radio("Spectra View", [280, 260])

# =====================
# PARSER
# =====================
def clean_filename(name):
    match = re.search(r"(\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})", name)
    return match.group(1) if match else name

def parse_file(file_bytes, filename):
    content = file_bytes.decode("utf-8", errors="replace").splitlines()

    spectra, wavelengths, ex_vals = {}, [], []

    for i, line in enumerate(content):
        if "excitation wavelength" in line.lower():
            ex_vals = [float(v) for v in line.split("\t")[2:] if v]

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

    matrix = np.array(matrix)

    for j, ex in enumerate(ex_vals):
        spectra[ex] = matrix[:, j]

    return {"wavelengths": np.array(wavelengths),
            "spectra": spectra,
            "filename": clean_filename(filename)}

# =====================
# FILE LOAD
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
# ✅ APIES DASHBOARD
# =====================
if page == "APIES Dashboard":

    st.title("APIES Dashboard")

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

        nearest = lambda v: np.argmin(np.abs(wl - v))

        mask_ir = (wl >= ir_start) & (wl <= ir_end)
        mask_if = (wl >= if_start) & (wl <= if_end)

        auc_ir = np.trapezoid(y[mask_ir], wl[mask_ir])
        auc_if = np.trapezoid(y[mask_if], wl[mask_if])

        irif_auc = auc_ir / auc_if if auc_if != 0 else np.nan
        i350_i330 = y[nearest(350)] / y[nearest(330)] if y[nearest(330)] != 0 else np.nan

        rows.append({
            "File": name,
            "Index": i,
            "IR/IF (AUC)": irif_auc,
            "I350/I330": i350_i330,
            "AUC IR": auc_ir,
            "AUC IF": auc_if,
            "Aggregation Index": np.nan,
            "Concentration": np.nan
        })

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True)

    # ✅ APIES FULL DASHBOARD
    st.subheader("APIES Dashboard Plot")

    fig = go.Figure()
    x = df["Index"].values

    def add_metric(y_vals, name):
        if len(x) > 1 and not np.all(np.isnan(y_vals)):
            coeffs = np.polyfit(x, y_vals, 1)
            fit = np.polyval(coeffs, x)

            ss_res = np.sum((y_vals - fit)**2)
            ss_tot = np.sum((y_vals - np.mean(y_vals))**2)
            r2 = 1 - ss_res/ss_tot if ss_tot != 0 else np.nan

            fig.add_trace(go.Scatter(
                x=x, y=y_vals,
                mode="lines+markers",
                name=f"{name} (R²={r2:.3f})"
            ))

            fig.add_trace(go.Scatter(
                x=x, y=fit,
                mode="lines",
                name=f"{name} Fit",
                line=dict(dash="dash")
            ))

    add_metric(df["IR/IF (AUC)"].values, "IR/IF")
    add_metric(df["I350/I330"].values, "I350/I330")
    add_metric(df["AUC IR"].values, "AUC IR")
    add_metric(df["AUC IF"].values, "AUC IF")

    st.plotly_chart(fig, use_container_width=True)

# =====================
# ✅ AUC ANALYSIS
# =====================
if page == "AUC Analysis":

    st.title("AUC Analysis")

    selected = st.selectbox("Dataset", list(data.keys()))
    d = data[selected]

    if ex_toggle not in d["spectra"]:
        st.warning("No data for this excitation")
        st.stop()

    wl = d["wavelengths"]
    y = d["spectra"][ex_toggle]

    start_wl = st.number_input("Start WL", value=float(min(wl)+20))
    end_wl = st.number_input("End WL", value=float(max(wl)-20))

    start_wl, end_wl = sorted([start_wl, end_wl])

    mask = (wl >= start_wl) & (wl <= end_wl)
    area = np.trapezoid(y[mask], wl[mask]) if np.any(mask) else 0

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=wl, y=y, name="Spectrum"))

    if np.any(mask):
        fig.add_trace(go.Scatter(
            x=wl[mask], y=y[mask],
            fill="tozeroy", name="AUC Region"
        ))

    st.plotly_chart(fig)
    st.metric("AUC", f"{area:.3f}")

    # ✅ Batch
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

            t = pd.to_datetime(name.split("_")[0], errors='coerce')

            results.append({"time": t, "AUC": auc_val})

        df_auc = pd.DataFrame(results).dropna().sort_values("time")

        st.dataframe(df_auc)

        # ✅ regression
        x = np.arange(len(df_auc))
        y_vals = df_auc["AUC"].values

        coeffs = np.polyfit(x, y_vals, 1)
        fit = np.polyval(coeffs, x)

        ss_res = np.sum((y_vals - fit)**2)
        ss_tot = np.sum((y_vals - np.mean(y_vals))**2)
        r2 = 1 - ss_res/ss_tot if ss_tot != 0 else np.nan

        fig_auc = go.Figure()

        fig_auc.add_trace(go.Scatter(
            x=df_auc["time"], y=y_vals,
            mode="lines+markers",
            name=f"AUC (R²={r2:.3f})"
        ))

        fig_auc.add_trace(go.Scatter(
            x=df_auc["time"], y=fit,
            mode="lines",
            name="Fit",
            line=dict(dash="dash")
        ))

        st.plotly_chart(fig_auc)
