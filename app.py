# v11.7 FULL RESTORE + AUC FIX
# Restores FULL Spectra + Kinetics pages (from working versions)
# Keeps AUC (fixed with np.trapezoid)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import io, zipfile, re

st.set_page_config(page_title='SpectraKinetics v11.7', layout='wide')

page = st.sidebar.radio("Navigation", ["Spectra Analysis", "Kinetics", "AUC Analysis"])

# CLEAN NAME

def clean_filename(name):
    match = re.search(r"(\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})", name)
    return match.group(1) if match else name[:15]

# PARSER (unchanged)
def parse_file(file_bytes, filename):
    content = file_bytes.decode('utf-8', errors='replace').splitlines()
    spectra = {}
    wavelengths = []
    ex_vals = []
    kinetics = None

    for i, line in enumerate(content):
        if "kinetic time" in line.lower():
            parts = line.split("	")
            blank_idx = next((j for j,p in enumerate(parts) if p.strip()==""), None)
            if blank_idx is not None:
                times = [float(x) for x in parts[blank_idx+1:] if x]
                wl, mat = [], []
                for row in content[i+1:]:
                    r = row.split("	")
                    try:
                        wl.append(float(r[1]))
                        mat.append([float(x) for x in r[blank_idx+1:blank_idx+1+len(times)]])
                    except:
                        continue
                kinetics = {"times": np.array(times), "wavelengths": np.array(wl), "matrix": np.array(mat)}

    for i, line in enumerate(content):
        parts = line.split("	")
        if 'excitation wavelength' in parts[0].lower():
            ex_vals = [float(v) for v in parts[2:] if v]
        if parts[0].isdigit():
            data_start = i
            break

    matrix = []
    for line in content[data_start:]:
        parts = line.split("	")
        try:
            wavelengths.append(float(parts[1]))
            matrix.append([float(x) for x in parts[2:2+len(ex_vals)]])
        except:
            continue

    matrix = np.array(matrix) if len(matrix)>0 else np.array([])
    for j, ex in enumerate(ex_vals if len(matrix)>0 else []):
        spectra[ex] = matrix[:, j]

    return {'wavelengths': np.array(wavelengths),'spectra': spectra,'filename': clean_filename(filename),'kinetics': kinetics}

# Upload
files = st.sidebar.file_uploader("Upload ≤200 files", type=['txt'], accept_multiple_files=True)
ex_toggle = st.sidebar.radio("Spectra View", [280, 260])

if files:
    st.session_state.datasets = {}
    for f in files[:200]:
        parsed = parse_file(f.read(), f.name)
        st.session_state.datasets[parsed['filename']] = parsed


data = st.session_state.get('datasets', {})
if not data:
    st.stop()

# =====================
# SPECTRA (RESTORED)
# =====================
if page == "Spectra Analysis":

    st.title("Spectra Analysis")

    rows = []

    for i,(name,d) in enumerate(data.items()):
        if 280 not in d['spectra']: continue

        wl = d['wavelengths']
        y = d['spectra'][280]

        ir_idx = np.argmax(y)
        ir_peak = wl[ir_idx]
        ir_int = y[ir_idx]

        mask = (wl>=300)&(wl<=390)
        if np.any(mask):
            y_if = y[mask]
            wl_if = wl[mask]
            idx = np.argmax(y_if)
            if_peak = wl_if[idx]
            if_int = y_if[idx]
        else:
            if_peak = np.nan
            if_int = np.nan

        nearest = lambda v: np.argmin(np.abs(wl-v))
        irif = y[nearest(280)]/y[nearest(340)] if y[nearest(340)]!=0 else np.nan
        pie = y[nearest(350)]/y[nearest(330)] if y[nearest(330)]!=0 else np.nan

        rows.append({
            "File":name,
            "Index":i,
            "IR/IF":irif,
            "I350/I330":pie,
            "Aggregation Index":np.nan,
            "Concentration (mg/mL)":np.nan,
            "IR (nm)":ir_peak,
            "IR Peak Intensity":ir_int,
            "IF (nm)":if_peak,
            "IF Peak Intensity":if_int
        })

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True)

    fig = go.Figure()
    for name,d in data.items():
        if ex_toggle in d['spectra']:
            fig.add_trace(go.Scatter(x=d['wavelengths'], y=d['spectra'][ex_toggle], name=name))

    st.plotly_chart(fig, use_container_width=True)

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=df['Index'], y=df['IR/IF'], name='IR/IF'))
    fig2.add_trace(go.Scatter(x=df['Index'], y=df['I350/I330'], name='I350/I330'))

    st.plotly_chart(fig2, use_container_width=True)

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

        st.plotly_chart(fig_k, use_container_width=True)

# =====================
# ✅ AUC (FIXED)
# =====================
if page == "AUC Analysis":

    st.title("AUC Analysis")

    selected_file = st.selectbox("Dataset", list(data.keys()))
    d = data[selected_file]

    wl = d['wavelengths']
    y = d['spectra'][ex_toggle]

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

    
if start_wl >= end_wl:
    st.warning("Start wavelength must be less than end wavelength")
    st.stop()
    
    mask = (wl >= start) & (wl <= end)
    area = np.trapezoid(y[mask], wl[mask]) if np.any(mask) else 0

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=wl, y=y, name='Full'))
    fig.add_trace(go.Scatter(x=wl[mask], y=y[mask], fill='tozeroy', name='Selected'))

    st.plotly_chart(fig, use_container_width=True)
    st.metric("AUC", f"{area:.3f}")

# EXPORT
st.sidebar.markdown("---")
