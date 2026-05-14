# v10, 14 May 2026
# =====================
# SPECTRA 
# =====================

# =====================
# IMPORTS
# =====================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import io, zipfile, re

# =====================
# CONFIG
# =====================
st.set_page_config(page_title='SpectraKinetics v11.7', layout='wide')

# =====================
# PAGE SELECTOR
# =====================
page = st.sidebar.radio(
    "Navigation",
    ["Spectra Analysis", "Kinetics", "AUC Analysis"]
)

# =====================
# CLEAN NAME FUNCTION
# =====================
def clean_filename(name):
    match = re.search(r"(\\d{4}-\\d{2}-\\d{2}-\\d{2}-\\d{2}-\\d{2})", name)
    return match.group(1) if match else name[:15]

# =====================
# PARSER FUNCTION
# =====================
def parse_file(file_bytes, filename):
    content = file_bytes.decode('utf-8', errors='replace').splitlines()

    spectra = {}
    wavelengths = []
    ex_vals = []
    kinetics = None

    # --- Kinetics ---
    for i, line in enumerate(content):
        if "kinetic time" in line.lower():
            parts = line.split("\t")
            blank_idx = next((j for j, p in enumerate(parts) if p.strip() == ""), None)

            if blank_idx is not None:
                times = [float(x) for x in parts[blank_idx+1:] if x]
                wl, mat = [], []

                for row in content[i+1:]:
                    r = row.split("\t")
                    try:
                        wl.append(float(r[1]))
                        mat.append([float(x) for x in r[blank_idx+1:blank_idx+1+len(times)]])
                    except:
                        continue

                kinetics = {
                    "times": np.array(times),
                    "wavelengths": np.array(wl),
                    "matrix": np.array(mat)
                }

    # --- Spectra ---
    for i, line in enumerate(content):
        parts = line.split("\t")

        if 'excitation wavelength' in parts[0].lower():
            ex_vals = [float(v) for v in parts[2:] if v]

        if parts[0].isdigit():
            data_start = i
            break

    matrix = []

    for line in content[data_start:]:
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
        'wavelengths': np.array(wavelengths),
        'spectra': spectra,
        'filename': clean_filename(filename),
        'kinetics': kinetics
    }

# =====================
# FILE UPLOAD
# =====================
files = st.sidebar.file_uploader(
    "Upload ≤200 files",
    type=['txt'],
    accept_multiple_files=True
)

ex_toggle = st.sidebar.radio("Spectra View", [280, 260])

if files:

    st.session_state.datasets = {}

    for i, f in enumerate(files[:200]):

        parsed = parse_file(f.read(), f.name)

        # ✅ Force unique names (CRITICAL FIX)
        unique_name = f"{parsed['filename']}_{i}"

        st.session_state.datasets[unique_name] = parsed

# =====================
# LOAD DATA
# =====================
data = st.session_state.get('datasets', {})

if not data:
    st.stop()

# =====================
# SPECTRA ANALYSIS (FULL FIXED SECTION)
# =====================
if page == "Spectra Analysis":

    st.title("Spectra Analysis")

    st.subheader("AUC Ratio Settings")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**IR Range (nm)**")
        ir_start = st.number_input("IR Start", value=270.0)
        ir_end = st.number_input("IR End", value=300.0)

    with col2:
        st.markdown("**IF Range (nm)**")
        if_start = st.number_input("IF Start", value=320.0)
        if_end = st.number_input("IF End", value=390.0)

    ir_start, ir_end = sorted([ir_start, ir_end])
    if_start, if_end = sorted([if_start, if_end])

    # =====================
    # TABLE
    # =====================
    rows = []

    for i, (name, d) in enumerate(data.items()):

        if 280 not in d['spectra']:
            continue

        wl = d['wavelengths']
        y = d['spectra'][280]

        ir_idx = np.argmax(y)
        ir_peak = wl[ir_idx]
        ir_int = y[ir_idx]

        mask = (wl >= 300) & (wl <= 390)

        if np.any(mask):
            y_if = y[mask]
            wl_if = wl[mask]
            idx = np.argmax(y_if)
            if_peak = wl_if[idx]
            if_int = y_if[idx]
        else:
            if_peak = np.nan
            if_int = np.nan

        nearest = lambda v: np.argmin(np.abs(wl - v))

        irif = y[nearest(280)] / y[nearest(340)] if y[nearest(340)] != 0 else np.nan
        pie = y[nearest(350)] / y[nearest(330)] if y[nearest(330)] != 0 else np.nan

        rows.append({
            "File": name,
            "Index": i,
            "IR/IF": irif,
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

    # =====================
    # ✅ FULL SPECTRA OVERLAY (RESTORED CORRECTLY)
    # =====================
    st.header(f"Spectra Overlay (Ex {ex_toggle})")

    fig = go.Figure()

    for name, d in data.items():

        if ex_toggle in d['spectra']:

            fig.add_trace(go.Scatter(
                x=d['wavelengths'],
                y=d['spectra'][ex_toggle],
                name=name
            ))

    if len(fig.data) == 0:
        st.warning("No spectra available for selected excitation.")
    else:
        fig.update_layout(
            xaxis_title="Wavelength (nm)",
            yaxis_title="Intensity",
            template="plotly_white"
        )

        st.plotly_chart(fig, use_container_width=True, key=f"spectra_{ex_toggle}")

    # =====================
    # ✅ APIES
    # =====================
    st.header("APIES (All Metrics Overlay)")

    fig2 = go.Figure()

    fig2.add_trace(go.Scatter(
        x=df['Index'],
        y=df['IR/IF'],
        name='IR/IF',
        mode='lines+markers'
    ))

    fig2.add_trace(go.Scatter(
        x=df['Index'],
        y=df['I350/I330'],
        name='I350/I330',
        mode='lines+markers'
    ))

    st.plotly_chart(fig2, use_container_width=True, key="apies_plot")

# =====================
# KINETICS (RESTORED)
# =====================
if page == "Kinetics":

    st.title("Kinetics Analysis")

    segments_280, segments_350 = [], []
    sorted_items = sorted(data.items(), key=lambda x: x[0])
    time_offset = 0

    for name, d in sorted_items:
        if d['kinetics'] is None:
            continue

        kin = d['kinetics']
        times = kin['times']

        if len(times) == 0:
            continue

        wl = kin['wavelengths']
        matrix = kin['matrix']

        if len(wl) == 0 or matrix.size == 0:
            continue

        idx_280 = np.argmin(np.abs(wl-280))
        idx_350 = np.argmin(np.abs(wl-350))

        signal_280 = matrix[idx_280,:]
        signal_350 = matrix[idx_350,:]

        shifted_time = times + time_offset

        segments_280.append((shifted_time, signal_280))
        segments_350.append((shifted_time, signal_350))

        time_offset += (times[-1] - times[0])

    if segments_280:
        fig_k = go.Figure()

        for t,y in segments_280:
            fig_k.add_trace(go.Scatter(x=t, y=y, name='280 nm'))
        for t,y in segments_350:
            fig_k.add_trace(go.Scatter(x=t, y=y, name='350 nm'))

        st.plotly_chart(fig_auc, use_container_width=True, key="auc_plot")

# =====================
# ✅ AUC (FIXED)
# =====================

# =====================
# AUC ANALYSIS
# =====================
if page == "AUC Analysis":

    st.title("AUC Analysis")

    # ---- SELECT DATASET ----
    selected_file = st.selectbox("Dataset", list(data.keys()))
    d = data[selected_file]

    # ---- EXTRACT DATA ----
    if ex_toggle not in d['spectra']:
        st.warning("Selected excitation not available.")
        st.stop()

    wl = d['wavelengths']
    y = d['spectra'][ex_toggle]

    # ---- LIMITS ----
    min_wl = float(np.min(wl))
    max_wl = float(np.max(wl))

    # ---- INPUT RANGE (THIS DEFINES start_wl / end_wl) ----
    st.subheader("Select Wavelength Range")

    col1, col2 = st.columns(2)

    with col1:
        start_wl = st.number_input(
            "Start Wavelength (nm)",
            min_value=min_wl,
            max_value=max_wl,
            value=float(min_wl + 20)
        )

    with col2:
        end_wl = st.number_input(
            "End Wavelength (nm)",
            min_value=min_wl,
            max_value=max_wl,
            value=float(max_wl - 20)
        )

    # ✅ fix ordering automatically
    start_wl, end_wl = sorted([start_wl, end_wl])

    # ---- SINGLE AUC ----
    mask = (wl >= start_wl) & (wl <= end_wl)

    if np.any(mask):
        area = np.trapezoid(y[mask], wl[mask])
    else:
        area = 0

    # ---- PLOT ----
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=wl, y=y, name="Full Spectrum"))

    fig.add_trace(go.Scatter(
        x=wl[mask],
        y=y[mask],
        fill='tozeroy',
        name="Selected Region"
    ))

    st.plotly_chart(fig, use_container_width=True, key="auc_single")

    st.metric("AUC", f"{area:.3f}")

    # =====================
    # ✅ BATCH AUC (THIS IS WHERE YOUR ERROR WAS)
    # =====================
    st.markdown("---")

# =====================
# ✅ BATCH AUC (FINAL CLEAN VERSION)
# =====================
st.markdown("---")

run_batch = st.button(
    "Calculate AUC for All Datasets",
    key="auc_batch_button"
)

if run_batch:

    results = []

    for name, dataset in data.items():

        if ex_toggle not in dataset['spectra']:
            continue

        wl_full = dataset['wavelengths']
        y_full = dataset['spectra'][ex_toggle]

        if len(wl_full) == 0:
            continue

        mask_full = (wl_full >= start_wl) & (wl_full <= end_wl)

        if not np.any(mask_full):
            continue

        auc_val = np.trapezoid(y_full[mask_full], wl_full[mask_full])

        try:
            timestamp = pd.to_datetime(name.split("_")[0])
        except:
            continue

        results.append({
            "time": timestamp,
            "AUC": auc_val,
            "file": name
        })

    if len(results) == 0:
        st.warning("No valid datasets for AUC calculation.")
    else:
        df_auc = pd.DataFrame(results).sort_values("time")

        fig_auc = go.Figure()

        fig_auc.add_trace(go.Scatter(
            x=df_auc["time"],
            y=df_auc["AUC"],
            mode="lines+markers",
            name="AUC over Time"
        ))

        st.plotly_chart(fig_auc, use_container_width=True, key="auc_batch_plot")

        st.subheader("AUC Results")
        st.dataframe(df_auc)

# ✅ Only run if inputs exist (CRITICAL FIX)
if run_batch:

    results = []

    for name, dataset in data.items():

        if ex_toggle not in dataset['spectra']:
            continue

        wl_full = dataset['wavelengths']
        y_full = dataset['spectra'][ex_toggle]

        if len(wl_full) == 0:
            continue

        mask_full = (wl_full >= start_wl) & (wl_full <= end_wl)

        if not np.any(mask_full):
            continue

        auc_val = np.trapezoid(y_full[mask_full], wl_full[mask_full])

        try:
            timestamp = pd.to_datetime(name.split("_")[0], format="%Y-%m-%d-%H-%M-%S")
        except:
            continue

        results.append({
            "time": timestamp,
            "AUC": auc_val,
            "file": name
        })

    if len(results) == 0:
        st.warning("No valid datasets for AUC calculation.")
    else:
        df_auc = pd.DataFrame(results).sort_values("time")

        fig_auc = go.Figure()

        fig_auc.add_trace(go.Scatter(
            x=df_auc["time"],
            y=df_auc["AUC"],
            mode="lines+markers",
            name="AUC over Time"
        ))

        st.plotly_chart(fig_auc, use_container_width=True, key="auc_batch_plot")

        st.subheader("AUC Results Table")
        st.dataframe(df_auc)

    # ---- FILE SELECTION ----
    selected_file = st.selectbox("Dataset", list(data.keys()))
    d = data[selected_file]

    # ✅ Check spectra exists
    if ex_toggle not in d['spectra']:
        st.warning("Selected excitation wavelength not available in this dataset.")
        st.stop()

    # ---- DATA EXTRACTION ----
    wl = d['wavelengths']
    y = d['spectra'][ex_toggle]

    # ✅ Ensure valid data
    if len(wl) == 0 or len(y) == 0:
        st.warning("No spectral data available.")
        st.stop()

    # ---- LIMITS ----
    min_wl = float(np.min(wl))
    max_wl = float(np.max(wl))

    # ---- INPUT UI ----
    st.subheader("Select Wavelength Range")

    col1, col2 = st.columns(2)

    with col1:
        start_wl = st.number_input(
            "Start Wavelength (nm)",
            min_value=min_wl,
            max_value=max_wl,
            value=float(min_wl + 20)
        )

    with col2:
        end_wl = st.number_input(
            "End Wavelength (nm)",
            min_value=min_wl,
            max_value=max_wl,
            value=float(max_wl - 20)
        )
# ✅ Ensure variables always exist (CRITICAL FIX)
start_wl = float(start_wl)
end_wl = float(end_wl)


    # ✅ safer handling (NO stop, just fix automatically)
    start_wl, end_wl = sorted([start_wl, end_wl])

    # ---- AUC ----
    mask = (wl >= start_wl) & (wl <= end_wl)

    if not np.any(mask):
        st.warning("Selected range contains no data.")
        st.stop()

    area = np.trapezoid(y[mask], wl[mask])

    # ---- PLOT ----
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=wl,
        y=y,
        name='Full Spectrum',
        line=dict(color='black')
    ))

    fig.add_trace(go.Scatter(
        x=wl[mask],
        y=y[mask],
        name='Selected Region',
        fill='tozeroy',
        line=dict(color='orange')
    ))

    fig.update_layout(
        title="AUC Selection",
        xaxis_title="Wavelength (nm)",
        yaxis_title="Intensity",
        template='plotly_white'
    )

    st.plotly_chart(fig, use_container_width=True)

    # ---- OUTPUT ----
    st.subheader("AUC Result")
    st.metric("Area Under Curve", f"{area:.3f}")
    st.info(f"Range: {start_wl:.1f} nm → {end_wl:.1f} nm")

st.markdown("---")

run_batch = st.button("Calculate AUC for All Datasets")

if run_batch:

    results = []

    for name, dataset in data.items():

        if ex_toggle not in dataset['spectra']:
            continue

        wl_full = dataset['wavelengths']
        y_full = dataset['spectra'][ex_toggle]

        if len(wl_full) == 0:
            continue

        mask_full = (wl_full >= start_wl) & (wl_full <= end_wl)

        if not np.any(mask_full):
            continue

        auc_val = np.trapezoid(y_full[mask_full], wl_full[mask_full])

        try:
            timestamp = pd.to_datetime(name, format="%Y-%m-%d-%H-%M-%S")
        except:
            continue

        results.append({
            "time": timestamp,
            "AUC": auc_val,
            "file": name
        })

    if len(results) == 0:
        st.warning("No valid datasets for AUC calculation.")

    else:
        df_auc = pd.DataFrame(results).sort_values("time")

        # ✅ REGRESSION SECTION (NOW SAFE)

        time_numeric = (df_auc["time"] - df_auc["time"].iloc[0]).dt.total_seconds()
        y = df_auc["AUC"].values

        coeffs = np.polyfit(time_numeric, y, 1)
        fit_line = np.polyval(coeffs, time_numeric)

        ss_res = np.sum((y - fit_line) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else np.nan

        # ✅ PLOT
        fig_auc = go.Figure()

        fig_auc.add_trace(go.Scatter(
            x=df_auc["time"],
            y=y,
            mode="lines+markers",
            name="AUC"
        ))

        fig_auc.add_trace(go.Scatter(
            x=df_auc["time"],
            y=fit_line,
            mode="lines",
            name=f"Linear Fit (R² = {r2:.4f})",
            line=dict(dash="dash")
        ))

        fig_auc.update_layout(
            title="AUC vs Time with Linear Fit",
            xaxis_title="Time",
            yaxis_title="AUC",
            template="plotly_white"
        )

        st.plotly_chart(fig_auc, use_container_width=True)

        # ✅ TABLE
        st.subheader("AUC Results Table")
        st.dataframe(df_auc)

        # ✅ R² DISPLAY
        st.metric("R² (Linear Fit)", f"{r2:.4f}")
