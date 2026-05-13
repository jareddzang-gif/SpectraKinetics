import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import io, zipfile, re

st.set_page_config(page_title='SpectraKinetics v9.6', layout='wide')

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

# PARSER

def parse_file(file_bytes, filename):
    content = file_bytes.decode('utf-8', errors='replace').splitlines()
    spectra, wavelengths, ex_vals = {}, [], []

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

    return {'wavelengths': np.array(wavelengths), 'spectra': spectra, 'filename': clean_filename(filename)}

# SIDEBAR
with st.sidebar:
    files = st.file_uploader("Upload ≤200 files", type=['txt'], accept_multiple_files=True)
    ex_toggle = st.radio("Spectra View", [280, 260])

    if files:
        st.session_state.datasets = {}
        for f in files[:200]:
            parsed = parse_file(f.read(), f.name)
            st.session_state.datasets[parsed['filename']] = parsed

st.title("SpectraKinetics v9.6 — Final Workflow")

data = st.session_state.datasets
if not data:
    st.stop()

# ANALYSIS
rows=[]

for i,(name,d) in enumerate(data.items()):
    if 280 not in d['spectra']: continue

    wl = d['wavelengths']
    y = d['spectra'][280]

    # IR peak
    ir_idx = np.argmax(y)
    ir_peak = wl[ir_idx]
    ir_int = y[ir_idx]

    # IF peak
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

    # ratios
    nearest = lambda v: np.argmin(np.abs(wl-v))
    irif = y[nearest(280)]/y[nearest(340)] if y[nearest(340)]!=0 else np.nan
    pie = y[nearest(350)]/y[nearest(330)] if y[nearest(330)]!=0 else np.nan

    # placeholders absorbance
    conc = np.nan
    agg = np.nan

    rows.append({
        "File":name,
        "Index":i,
        "IR/IF":irif,
        "I350/I330":pie,
        "Aggregation Index":agg,
        "Concentration":conc,
        "IR Peak (nm)":ir_peak,
        "IR Peak Intensity":ir_int,
        "IF Peak (nm)":if_peak,
        "IF Peak Intensity":if_int
    })

# DF
df = pd.DataFrame(rows)
st.dataframe(df, use_container_width=True)

# SPECTRA VIEW BASED ON TOGGLE
st.header(f"Spectra Overlay (Ex {ex_toggle} nm)")
fig = go.Figure()
for name,d in data.items():
    if ex_toggle in d['spectra']:
        fig.add_trace(go.Scatter(x=d['wavelengths'], y=d['spectra'][ex_toggle], name=name))
st.plotly_chart(fig, use_container_width=True)

# COMBINED METRICS GRAPH
st.header("Combined Metrics (Overlay)")
fig2 = go.Figure()

fig2.add_trace(go.Scatter(x=df['Index'], y=df['IR/IF'], name='IR/IF', mode='lines+markers'))
fig2.add_trace(go.Scatter(x=df['Index'], y=df['I350/I330'], name='I350/I330', mode='lines+markers'))

if df['Aggregation Index'].notna().any():
    fig2.add_trace(go.Scatter(x=df['Index'], y=df['Aggregation Index'], name='Aggregation Index', mode='lines+markers', yaxis='y2'))

if df['Concentration'].notna().any():
    fig2.add_trace(go.Scatter(x=df['Index'], y=df['Concentration'], name='Concentration', mode='lines+markers', yaxis='y2'))

fig2.update_layout(
    yaxis=dict(title="Fluorescence Ratios"),
    yaxis2=dict(title="Absorbance Metrics", overlaying='y', side='right'),
    title="All Key Metrics"
)

st.plotly_chart(fig2, use_container_width=True)

# EXPORT
st.header("Export")

def build_zip():
    buf=io.BytesIO()
    with zipfile.ZipFile(buf,'w') as z:
        z.writestr('analysis.csv', df.to_csv(index=False))
    buf.seek(0)
    return buf

if st.button('Build ZIP'):
    st.download_button('Download ZIP', build_zip(), 'spectrakinetics_v9_6.zip')
