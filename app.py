# v13 FINAL VERIFIED BUILD

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import re, uuid

st.set_page_config(page_title='SpectraKinetics v13', layout='wide')

page = st.sidebar.radio("Navigation", ["Spectra Analysis", "Kinetics", "AUC Analysis"])
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
        "filename": clean_filename(filename)
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
    st.info("Upload files to begin.")
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

        nearest = lambda v: np.argmin(np.abs(wl - v))

        # AUC
        mask_ir = (wl >= ir_start) & (wl <= ir_end)
        mask_if = (wl >= if_start) & (wl <= if_end)

        auc_ir = np.trapezoid(y[mask_ir], wl[mask_ir]) if np.any(mask_ir) else np.nan
        auc_if = np.trapezoid(y[mask_if], wl[mask_if]) if np.any(mask_if) else np.nan

        irif_auc = auc_ir / auc_if if (not np.isnan(auc_if) and auc_if != 0) else np.nan
        irif_point = y[nearest(280)] / y[nearest(340)] if y[nearest(340)] != 0 else np.nan
        pie = y[nearest(350)] / y[nearest(330)] if y[nearest(330)] != 0 else np.nan

        ir_idx = np.argmax(y)
        ir_peak = wl[ir_idx]
        ir_int = y[ir_idx]

        mask_peak = (wl >= 300) & (wl <= 390)
        if np.any(mask_peak):
            y_if = y[mask_peak]
            wl_if = wl[mask_peak]
            idx = np.argmax(y_if)
            if_peak = wl_if[idx]
            if_int = y_if[idx]
        else:
            if_peak, if_int = np.nan, np.nan

        rows.append({
            "File": name,
            "Index": i,
            "IR/IF": irif_point,
            "IR/IF (AUC)": irif_auc,
            "AUC IR": auc_ir,
            "AUC IF": auc_if,
            "I350/I330": pie,
            "Aggregation Index": np.nan,
            "Concentration (mg/mL)": np.nan,
            "IR (nm)": ir_peak,
            "IR Peak Intensity": ir_int,
            "IF (nm)": if_peak,
            "IF Peak Intensity": if_int
        })

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True)

    # Spectra overlay
    fig = go.Figure()
    for name, d in data.items():
        if ex_toggle in d["spectra"]:
            fig.add_trace(go.Scatter(
                x=d["wavelengths"],
                y=d["spectra"][ex_toggle],
                name=name
            ))
    st.plotly_chart(fig, use_container_width=True)

    # APIES
    st.subheader("APIES (IR/IF AUC with Regression)")
    fig2 = go.Figure()

    x_vals = df["Index"].values
    y_vals = df["IR/IF (AUC)"].values

    if len(df) > 1:
        coeffs = np.polyfit(x_vals, y_vals, 1)
        fit = np.polyval(coeffs, x_vals)

        ss_res = np.sum((y_vals - fit) ** 2)
        ss_tot = np.sum((y_vals - np.mean(y_vals)) ** 2)
        r2 = 1 - (ss_res / ss_tot)

        fig2.add_trace(go.Scatter(x=x_vals, y=y_vals, mode="lines+markers",
                                 name=f"IR/IF (R²={r2:.3f})"))
        fig2.add_trace(go.Scatter(x=x_vals, y=fit, mode="lines",
                                 name="Fit", line=dict(dash="dash")))

    st.plotly_chart(fig2)

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

    start_wl = st.number_input("Start WL", value=float(min(wl) + 20))
    end_wl = st.number_input("End WL", value=float(max(wl) - 20))

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

    # ✅ BATCH AUC WITH REGRESSION
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
                t = name

            results.append({"time": t, "AUC": auc_val})

        df_auc = pd.DataFrame(results).sort_values("time")

        st.dataframe(df_auc)

        # regression
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
            name="AUC"
        ))

        fig_auc.add_trace(go.Scatter(
            x=df_auc["time"], y=fit,
            mode="lines",
            name=f"Fit (R²={r2:.3f})",
            line=dict(dash="dash")
        ))

        st.plotly_chart(fig_auc)
