import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import io, zipfile, re

st.set_page_config(page_title='SpectraKinetics v10 (Kinetics Fixed)', layout='wide')

# LOGO
st.markdown("""
<div style='text-align:center;'>
<h1 style='color:#0B3D91; font-size:60px;'>NBL</h1>
</div>
""", unsafe_allow_html=True)

if 'datasets' not in st.session_state:
    st.session_state.datasets = {}

# CLEAN NAME

def clean_filename(name):
    match = re.search(r"(\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})", name)
    return match.group(1) if match else name[:15]

# PARSER WITH KINETICS

def parse_file(file_bytes, filename):
    content = file_bytes.decode('utf-8', errors='replace').splitlines()

    spectra = {}
    wavelengths = []
    ex_vals = []
    kinetics = None

    for i, line in enumerate(content):
        if "kinetic time" in line.lower():
            parts = line.split("	")

            blank_idx = None
            for j, p in enumerate(parts):
                if p.strip() == "":
                    blank_idx = j
                    break

            if blank_idx is not None:
                time_vals = [float(x) for x in parts[blank_idx+1:] if x]
                wl = []
                matrix = []

                for row in content[i+1:]:
                    r = row.split("	")
                    try:
                        wl.append(float(r[1]))
                        matrix.append([float(x) for x in r[blank_idx+1:blank_idx+1+len(time_vals)]])
                    except:
                        continue

                kinetics = {
                    "times": np.array(time_vals),
                    "wavelengths": np.array(wl),
                    "matrix": np.array(matrix)
                }

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

    matrix = np.array(matrix)
    for j, ex in enumerate(ex_vals):
        spectra[ex] = matrix[:, j]

    return {
        'wavelengths': np.array(wavelengths),
        'spectra': spectra,
        'filename': clean_filename(filename),
        'kinetics': kinetics
    }

# SIDEBAR
with st.sidebar:
    files = st.file_uploader("Upload ≤200 files", type=['txt'], accept_multiple_files=True)
    ex_toggle = st.radio("Spectra View", [280, 260])

    if files:
        st.session_state.datasets = {}
        for f in files[:200]:
            parsed = parse_file(f.read(), f.name)
            st.session_state.datasets[parsed['filename']] = parsed

st.title("SpectraKinetics v10 — Kinetics Enabled (Stable)")

data = st.session_state.datasets
if not data:
    st.stop()

# ANALYSIS
rows=[]

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
        "IR Peak (nm)":ir_peak,
        "IR Peak Intensity":ir_int,
        "IF Peak (nm)":if_peak,
        "IF Peak Intensity":if_int
    })


df = pd.DataFrame(rows)
st.dataframe(df, use_container_width=True)

# SPECTRA
st.header(f"Spectra Overlay (Ex {ex_toggle} nm)")
fig = go.Figure()
for name,d in data.items():
    if ex_toggle in d['spectra']:
        fig.add_trace(go.Scatter(x=d['wavelengths'], y=d['spectra'][ex_toggle], name=name))
st.plotly_chart(fig, use_container_width=True, key=f"spectra_{ex_toggle}")

# KINETICS
st.header("Kinetics Viewer")

for name, d in data.items():
    if d['kinetics'] is not None:
        kin = d['kinetics']

        st.subheader(f"Kinetics: {name}")

        times = kin['times']
        wl = kin['wavelengths']
        matrix = kin['matrix']

        idx = np.argmin(np.abs(wl-350))
        signal = matrix[idx, :]

        fig_k = go.Figure()
        fig_k.add_trace(go.Scatter(x=times, y=signal, mode='lines'))
        fig_k.update_layout(
            title="Intensity vs Time (350 nm)",
            xaxis_title="Time (s)",
            yaxis_title="Intensity"
        )

        st.plotly_chart(fig_k, use_container_width=True, key=f"kinetics_{name}")

# EXPORT
st.header("Export")

def build_zip():
    buf=io.BytesIO()
    with zipfile.ZipFile(buf,'w') as z:
        z.writestr('analysis.csv', df.to_csv(index=False))
    buf.seek(0)
    return buf

if st.button('Build ZIP'):
    st.download_button('Download ZIP', build_zip(), 'spectrakinetics_v10_fixed.zip')
