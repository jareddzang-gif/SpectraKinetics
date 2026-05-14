# v15 FINAL – APIES MULTI-AXIS + OVERLAY + CLEAN AUC

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import re

st.set_page_config(page_title='SpectraKinetics v15', layout='wide')

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

        t = pd.to_datetime(name.split("_")[0], errors='coerce')

        rows.append({
            "File": name,
            "Time": t,
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

    # =====================
    # ✅ FULL SPECTRA OVERLAY (RESTORED)
    # =====================
    st.subheader("Spectra Overlay")

    fig_spec = go.Figure()

    for name, d in data.items():
        if ex_toggle in d["spectra"]:
            fig_spec.add_trace(go.Scatter(
                x=d["wavelengths"],
                y=d["spectra"][ex_toggle],
                name=name
            ))

    fig_spec.update_layout(
        xaxis_title="Wavelength",
        yaxis_title="Intensity"
    )

    st.plotly_chart(fig_spec, use_container_width=True)

    # =====================
    # ✅ MULTI-AXIS APIES DASHBOARD
    # =====================
    st.subheader("APIES Multi-Metric Dashboard")

    df = df.sort_values("Time")

    x_vals = df["Time"].fillna(df["Index"])

    fig = go.Figure()

    # ✅ LEFT AXIS (PRIMARY)
    fig.add_trace(go.Scatter(
        x=x_vals,
        y=df["IR/IF (AUC)"],
        name="IR/IF",
        mode="lines+markers",
        yaxis="y1"
    ))

    # ✅ RIGHT AXIS (SECONDARY)
    metrics_right = [
        ("I350/I330", df["I350/I330"]),
        ("AUC IR", df["AUC IR"]),
        ("AUC IF", df["AUC IF"])
    ]

    for name, series in metrics_right:
        fig.add_trace(go.Scatter(
            x=x_vals,
            y=series,
            name=name,
            mode="lines+markers",
            yaxis="y2"
        ))

    fig.update_layout(
        xaxis=dict(title="Time / Sample"),
        yaxis=dict(title="IR/IF (AUC)", side="left"),
        yaxis2=dict(title="Other Metrics",
                    overlaying='y',
                    side='right'),
        template="plotly_white"
    )

    st.plotly_chart(fig, use_container_width=True)

# =====================
# ✅ AUC ANALYSIS (FINAL CLEAN)
# =====================
if page == "AUC Analysis":

    st.title("AUC Analysis")

    selected = st.selectbox("Dataset", list(data.keys()))
    d = data[selected]

    if ex_toggle not in d["spectra"]:
        st.warning("No excitation data")
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
            fill="tozeroy",
            name="AUC Region"
        ))

    st.plotly_chart(fig)
    st.metric("AUC", f"{area:.3f}")

    # ✅ BATCH FIXED
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
            if pd.isna(t):
                continue

            results.append({"time": t, "AUC": auc_val})

        df_auc = (
            pd.DataFrame(results)
            .groupby("time", as_index=False)
            .mean()
            .sort_values("time")
        )

        st.dataframe(df_auc)

        x = np.arange(len(df_auc))
        y_vals = df_auc["AUC"].values

        coeffs = np.polyfit(x, y_vals, 1)
        fit = np.polyval(coeffs, x)

        ss_res = np.sum((y_vals - fit)**2)
        ss_tot = np.sum((y_vals - np.mean(y_vals))**2)
        r2 = 1 - ss_res/ss_tot if ss_tot != 0 else np.nan

        fig_auc = go.Figure()
        fig_auc.add_trace(go.Scatter(
            x=df_auc["time"],
            y=y_vals,
            name=f"AUC (R²={r2:.3f})",
            mode="lines+markers"
        ))

        fig_auc.add_trace(go.Scatter(
            x=df_auc["time"],
            y=fit,
            name="Fit",
            line=dict(dash="dash")
        ))

        st.plotly_chart(fig_auc, use_container_width=True)
