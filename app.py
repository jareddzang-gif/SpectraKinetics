# v10, 14 May 2026

# =====================
# SPECTRA ANALYSIS (FINAL CLEAN VERSION)
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

if page == "Spectra Analysis":

    st.title("Spectra Analysis")

    st.subheader("AUC Ratio Settings")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**IR Range (nm)**")
        ir_start = st.number_input("IR Start", value=270.0, key="ir_start")
        ir_end = st.number_input("IR End", value=300.0, key="ir_end")

    with col2:
        st.markdown("**IF Range (nm)**")
        if_start = st.number_input("IF Start", value=320.0, key="if_start")
        if_end = st.number_input("IF End", value=390.0, key="if_end")

    ir_start, ir_end = sorted([ir_start, ir_end])
    if_start, if_end = sorted([if_start, if_end])

    # =====================
    # TABLE (WITH AUC)
    # =====================
    rows = []

    for i, (name, d) in enumerate(data.items()):

        if 280 not in d['spectra']:
            continue

        wl = d['wavelengths']
        y = d['spectra'][280]

        # ---- IR peak ----
        ir_idx = np.argmax(y)
        ir_peak = wl[ir_idx]
        ir_int = y[ir_idx]

        # ---- IF peak ----
        mask_peak = (wl >= 300) & (wl <= 390)

        if np.any(mask_peak):
            y_if = y[mask_peak]
            wl_if = wl[mask_peak]
            idx = np.argmax(y_if)
            if_peak = wl_if[idx]
            if_int = y_if[idx]
        else:
            if_peak = np.nan
            if_int = np.nan

        # =====================
        # ✅ AUC CALCULATIONS
        # =====================
        mask_ir = (wl >= ir_start) & (wl <= ir_end)
        auc_ir = np.trapezoid(y[mask_ir], wl[mask_ir]) if np.any(mask_ir) else np.nan

        mask_if_auc = (wl >= if_start) & (wl <= if_end)
        auc_if = np.trapezoid(y[mask_if_auc], wl[mask_if_auc]) if np.any(mask_if_auc) else np.nan

        irif = auc_ir / auc_if if (not np.isnan(auc_if) and auc_if != 0) else np.nan

        # ---- original ratio preserved ----
        nearest = lambda v: np.argmin(np.abs(wl - v))
        pie = y[nearest(350)] / y[nearest(330)] if y[nearest(330)] != 0 else np.nan

        rows.append({
            "File": name,
            "Index": i,
            "IR/IF": irif,
            "IR/IF (AUC)": irif,
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
    st.session_state["spectra_df"] = df
    st.dataframe(df, use_container_width=True)

    # =====================
    # ✅ FULL SPECTRA OVERLAY
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

    import uuid

    if len(fig.data) == 0:
        st.warning("No spectra available for selected excitation.")
    else:
        fig.update_layout(
            xaxis_title="Wavelength (nm)",
            yaxis_title="Intensity",
            template="plotly_white"
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            key=f"spectra_overlay_{uuid.uuid4()}"
        )

# =====================
# ✅ APIES WITH REGRESSION (FINAL CLEAN)
# =====================

# ✅ SAFE: ensure df exists ONLY if we're in Spectra context
try:
    df = pd.DataFrame(rows)
except:
    st.stop()

st.header("APIES (All Metrics Overlay)")

fig2 = go.Figure()

# ---- X axis (sample order) ----
x_vals = df['Index'].values

# =====================
# ✅ IR/IF (AUC)
# =====================
y_irif = df['IR/IF'].values

if len(x_vals) > 1 and not np.all(np.isnan(y_irif)):
    coeffs = np.polyfit(x_vals, y_irif, 1)
    fit_irif = np.polyval(coeffs, x_vals)

    ss_res = np.sum((y_irif - fit_irif) ** 2)
    ss_tot = np.sum((y_irif - np.mean(y_irif)) ** 2)
    r2_irif = 1 - (ss_res / ss_tot) if ss_tot != 0 else np.nan

    fig2.add_trace(go.Scatter(
        x=x_vals,
        y=y_irif,
        mode='lines+markers',
        name=f'IR/IF (R²={r2_irif:.3f})'
    ))

    fig2.add_trace(go.Scatter(
        x=x_vals,
        y=fit_irif,
        mode='lines',
        name='IR/IF Fit',
        line=dict(dash='dash')
    ))

# =====================
# ✅ I350/I330
# =====================
y_pie = df['I350/I330'].values

if len(x_vals) > 1 and not np.all(np.isnan(y_pie)):
    coeffs = np.polyfit(x_vals, y_pie, 1)
    fit_pie = np.polyval(coeffs, x_vals)

    ss_res = np.sum((y_pie - fit_pie) ** 2)
    ss_tot = np.sum((y_pie - np.mean(y_pie)) ** 2)
    r2_pie = 1 - (ss_res / ss_tot) if ss_tot != 0 else np.nan

    fig2.add_trace(go.Scatter(
        x=x_vals,
        y=y_pie,
        mode='lines+markers',
        name=f'I350/I330 (R²={r2_pie:.3f})'
    ))

    fig2.add_trace(go.Scatter(
        x=x_vals,
        y=fit_pie,
        mode='lines',
        name='I350/I330 Fit',
        line=dict(dash='dash')
    ))

# =====================
# ✅ FINAL PLOT
# =====================
fig2.update_layout(
    title="APIES with Linear Regression",
    xaxis_title="Sample Index",
    yaxis_title="Metric Value",
    template="plotly_white"
)

st.plotly_chart(fig2, use_container_width=True, key="apies_regression")

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
# ✅ AUC ANALYSIS (FINAL CLEAN VERSION)
# =====================
if page == "AUC Analysis":

    st.title("AUC Analysis")

    # ---- SELECT DATASET ----
    selected_file = st.selectbox(
        "Dataset",
        list(data.keys()),
        key="auc_dataset_select"
    )

    d = data[selected_file]

    # ---- VALIDATION ----
    if ex_toggle not in d['spectra']:
        st.warning("Selected excitation not available.")
        st.stop()

    wl = d['wavelengths']
    y = d['spectra'][ex_toggle]

    if len(wl) == 0:
        st.warning("No wavelength data available.")
        st.stop()

    # ---- LIMITS ----
    min_wl = float(np.min(wl))
    max_wl = float(np.max(wl))

    # =====================
    # ✅ INPUT RANGE
    # =====================
    st.subheader("Select Wavelength Range")

    col1, col2 = st.columns(2)

    with col1:
        start_wl = st.number_input(
            "Start Wavelength (nm)",
            min_value=min_wl,
            max_value=max_wl,
            value=float(min_wl + 20),
            key="start_wl_input"
        )

    with col2:
        end_wl = st.number_input(
            "End Wavelength (nm)",
            min_value=min_wl,
            max_value=max_wl,
            value=float(max_wl - 20),
            key="end_wl_input"
        )

    # ✅ ensure correct ordering
    start_wl, end_wl = sorted([float(start_wl), float(end_wl)])

# =====================
# ✅ SINGLE DATASET AUC + VISUALIZATION (CORRECT ORDER)
# =====================

# ✅ Ensure inputs ALWAYS exist before use
start_wl = float(start_wl)
end_wl = float(end_wl)
start_wl, end_wl = sorted([start_wl, end_wl])

# ✅ Now safe to calculate
mask = (wl >= start_wl) & (wl <= end_wl)

if np.any(mask):
    area = np.trapezoid(y[mask], wl[mask])
else:
    area = 0

# ✅ Plot full spectrum + selected region
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=wl,
    y=y,
    name="Full Spectrum",
    line=dict(color='black')
))

if np.any(mask):
    fig.add_trace(go.Scatter(
        x=wl[mask],
        y=y[mask],
        fill='tozeroy',
        name="Selected Region (AUC)",
        line=dict(color='orange')
    ))

fig.update_layout(
    title=f"AUC Selection ({start_wl:.1f}–{end_wl:.1f} nm)",
    xaxis_title="Wavelength (nm)",
    yaxis_title="Intensity",
    template="plotly_white"
)

st.plotly_chart(fig, use_container_width=True, key="auc_single_plot")

# ✅ Outputs
st.metric("AUC", f"{area:.3f}")
st.info(f"Range: {start_wl:.1f} nm → {end_wl:.1f} nm")

    # =====================
    # ✅ BATCH AUC
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

        # ✅ regression
        time_numeric = (df_auc["time"] - df_auc["time"].iloc[0]).dt.total_seconds()
        y_vals = df_auc["AUC"].values

        coeffs = np.polyfit(time_numeric, y_vals, 1)
        fit_line = np.polyval(coeffs, time_numeric)

        ss_res = np.sum((y_vals - fit_line) ** 2)
        ss_tot = np.sum((y_vals - np.mean(y_vals)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else np.nan

        # ✅ plot
        fig_auc = go.Figure()

        fig_auc.add_trace(go.Scatter(
            x=df_auc["time"],
            y=y_vals,
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
            title="AUC vs Time",
            xaxis_title="Time",
            yaxis_title="AUC",
            template="plotly_white"
        )

        # ✅ FIXED plotting placement
        import uuid

        st.plotly_chart(
            fig_auc,
            use_container_width=True,
            key=f"auc_batch_plot_{uuid.uuid4()}"
        )

        st.subheader("AUC Results Table")
        st.dataframe(df_auc)

        st.metric("R² (Linear Fit)", f"{r2:.4f}")
