# v10, 14 May 2026

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import io, zipfile, re, uuid

st.set_page_config(page_title='SpectraKinetics v11.7', layout='wide')

page = st.sidebar.radio(
    "Navigation",
    ["Spectra Analysis", "Kinetics", "AUC Analysis"]
)

def clean_filename(name):
    match = re.search(r"(\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})", name)
    return match.group(1) if match else name[:15]

def parse_file(file_bytes, filename):
    content = file_bytes.decode('utf-8', errors='replace').splitlines()

    spectra, wavelengths, ex_vals, kinetics = {}, [], [], None

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

files = st.sidebar.file_uploader("Upload files", type=['txt'], accept_multiple_files=True)
ex_toggle = st.sidebar.radio("Spectra View", [280, 260])

if files:
    st.session_state.datasets = {}
    for i, f in enumerate(files):
        parsed = parse_file(f.read(), f.name)
        st.session_state.datasets[f"{parsed['filename']}_{i}"] = parsed

data = st.session_state.get('datasets', {})
if not data:
    st.stop()

# =====================
# ✅ SPECTRA ANALYSIS
# =====================
if page == "Spectra Analysis":

    st.title("Spectra Analysis")

    ir_start = st.number_input("IR Start", value=270.0)
    ir_end = st.number_input("IR End", value=300.0)

    if_start = st.number_input("IF Start", value=320.0)
    if_end = st.number_input("IF End", value=390.0)

    ir_start, ir_end = sorted([ir_start, ir_end])
    if_start, if_end = sorted([if_start, if_end])

    rows = []

    for i, (name, d) in enumerate(data.items()):
        if 280 not in d['spectra']:
            continue

        wl = d['wavelengths']
        y = d['spectra'][280]

        mask_ir = (wl >= ir_start) & (wl <= ir_end)
        mask_if = (wl >= if_start) & (wl <= if_end)

        auc_ir = np.trapezoid(y[mask_ir], wl[mask_ir]) if np.any(mask_ir) else np.nan
        auc_if = np.trapezoid(y[mask_if], wl[mask_if]) if np.any(mask_if) else np.nan

        irif = auc_ir / auc_if if auc_if not in [0, np.nan] else np.nan

        rows.append({"File": name, "Index": i, "IR/IF": irif, "AUC IR": auc_ir, "AUC IF": auc_if})

    df = pd.DataFrame(rows)
    st.dataframe(df)

    fig = go.Figure()
    for name, d in data.items():
        if ex_toggle in d['spectra']:
            fig.add_trace(go.Scatter(x=d['wavelengths'], y=d['spectra'][ex_toggle], name=name))

    st.plotly_chart(fig, use_container_width=True, key=f"spectra_{uuid.uuid4()}")

    # ✅ APIES
    fig2 = go.Figure()
    x_vals = df['Index']

    if len(df) > 1:
        y = df['IR/IF']
        coeffs = np.polyfit(x_vals, y, 1)
        fit = np.polyval(coeffs, x_vals)

        r2 = 1 - np.sum((y-fit)**2)/np.sum((y-np.mean(y))**2)

        fig2.add_trace(go.Scatter(x=x_vals, y=y, mode="lines+markers"))
        fig2.add_trace(go.Scatter(x=x_vals, y=fit, mode="lines", name=f"Fit R²={r2:.3f}"))

    st.plotly_chart(fig2)

# =====================
# ✅ KINETICS
# =====================
if page == "Kinetics":

    fig_k = go.Figure()

    for name, d in data.items():
        if d['kinetics'] is None:
            continue

        kin = d['kinetics']
        wl = kin['wavelengths']
        mat = kin['matrix']

        idx = np.argmin(np.abs(wl-280))
        fig_k.add_trace(go.Scatter(x=kin['times'], y=mat[idx,:], name=name))

    st.plotly_chart(fig_k)

# =====================
# ✅ AUC ANALYSIS
# =====================
if page == "AUC Analysis":

    st.title("AUC Analysis")

    selected_file = st.selectbox("Dataset", list(data.keys()))
    d = data[selected_file]

    if ex_toggle not in d['spectra']:
        st.warning("Excitation not found")
        st.stop()

    wl = d['wavelengths']
    y = d['spectra'][ex_toggle]

    min_wl, max_wl = float(np.min(wl)), float(np.max(wl))

    start_wl = st.number_input("Start WL", min_value=min_wl, max_value=max_wl, value=min_wl+20)
    end_wl = st.number_input("End WL", min_value=min_wl, max_value=max_wl, value=max_wl-20)

    start_wl, end_wl = sorted([start_wl, end_wl])

    mask = (wl >= start_wl) & (wl <= end_wl)
    area = np.trapezoid(y[mask], wl[mask]) if np.any(mask) else 0

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=wl, y=y))
    if np.any(mask):
        fig.add_trace(go.Scatter(x=wl[mask], y=y[mask], fill="tozeroy"))

    st.plotly_chart(fig)
    st.metric("AUC", f"{area:.3f}")

    if st.button("Calculate AUC for All Datasets"):

        results = []
        for name, dataset in data.items():
            if ex_toggle not in dataset['spectra']:
                continue

            wl_full = dataset['wavelengths']
            y_full = dataset['spectra'][ex_toggle]

            mask = (wl_full >= start_wl) & (wl_full <= end_wl)
            if not np.any(mask):
                continue

            val = np.trapezoid(y_full[mask], wl_full[mask])
            results.append({"file": name, "AUC": val})

        df_auc = pd.DataFrame(results)
        st.dataframe(df_auc)
``
