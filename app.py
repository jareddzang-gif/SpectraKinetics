from plotly.subplots import make_subplots
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import io, zipfile
import re

st.set_page_config(page_title='SpectraKinetics v9.5', layout='wide')

# ---------------- LOGO ----------------
st.markdown("""
<div style='text-align:center;'>
<h1 style='color:#0B3D91; font-size:60px;'>NBL</h1>
</div>
""", unsafe_allow_html=True)

if 'datasets' not in st.session_state:
    st.session_state.datasets = {}

# ---------------- RENAME FILE ----------------
def clean_filename(name):
    match = re.search(r"(\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})", name)
    return match.group(1) if match else name[:15]

# ---------------- PARSER ----------------
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


def nearest(arr, val):
    return np.argmin(np.abs(arr - val))

# SIDEBAR
with st.sidebar:
    files = st.file_uploader("Upload ≤200 files", type=['txt'], accept_multiple_files=True)

    if files:
        st.session_state.datasets = {}
        for f in files[:200]:
            parsed = parse_file(f.read(), f.name)
            st.session_state.datasets[parsed['filename']] = parsed

st.title("SpectraKinetics v9.5 — Peak Analysis Upgrade")

data = st.session_state.datasets
if not data:
    st.stop()

# ANALYSIS
rows=[]
spectra280=[]
spectra260=[]

for i,(name,d) in enumerate(data.items()):
    if 280 not in d['spectra']: continue

    wl = d['wavelengths']
    y280 = d['spectra'][280]

    # IR (Rayleigh/Mie region peak - global max)
    ir_idx = np.argmax(y280)
    ir_peak_wl = wl[ir_idx]
    ir_peak_intensity = y280[ir_idx]

    # IF (fluorescence region 300–390)
    mask_if = (wl >= 300) & (wl <= 390)
    if np.any(mask_if):
        y_if = y280[mask_if]
        wl_if = wl[mask_if]
        if_idx_local = np.argmax(y_if)
        if_peak_wl = wl_if[if_idx_local]
        if_peak_intensity = y_if[if_idx_local]
    else:
        if_peak_wl = np.nan
        if_peak_intensity = np.nan

    # Ratios (unchanged)
    i280 = nearest(wl,280)
    i340 = nearest(wl,340)
    i350 = nearest(wl,350)
    i330 = nearest(wl,330)

    irif = y280[i280]/y280[i340] if y280[i340]!=0 else np.nan
    pie = y280[i350]/y280[i330] if y280[i330]!=0 else np.nan

    rows.append({
        "File":name,
        "Index":i,
        "IR/IF":irif,
        "I350/I330":pie,
        "IR Peak (nm)":ir_peak_wl,
        "IR Peak Intensity":ir_peak_intensity,
        "IF Peak (nm)":if_peak_wl,
        "IF Peak Intensity":if_peak_intensity
    })

    spectra280.append(y280)
    if 260 in d['spectra']:
        spectra260.append(d['spectra'][260])


df = pd.DataFrame(rows)
st.dataframe(df, use_container_width=True)

# -------- OVERLAY SPECTRA 280 --------
st.header("Spectra Overlay (280 nm)")
fig280 = go.Figure()
for name,d in data.items():
    if 280 in d['spectra']:
        fig280.add_trace(go.Scatter(x=d['wavelengths'], y=d['spectra'][280], name=name))
st.plotly_chart(fig280, use_container_width=True)

# -------- OVERLAY SPECTRA 260 --------
st.header("Spectra Overlay (260 nm)")
fig260 = go.Figure()
has260=False
for name,d in data.items():
    if 260 in d['spectra']:
        has260=True
        fig260.add_trace(go.Scatter(x=d['wavelengths'], y=d['spectra'][260], name=name))
if has260:
    st.plotly_chart(fig260, use_container_width=True)
else:
    st.info("No 260 nm spectra found")

# EXPORT
st.header("Export")

def build_zip():
    buf=io.BytesIO()
    with zipfile.ZipFile(buf,'w') as z:
        z.writestr('analysis.csv', df.to_csv(index=False))
    buf.seek(0)
    return buf

if st.button('Build ZIP'):
    st.download_button('Download ZIP', build_zip(), 'spectrakinetics_v9_5.zip')
