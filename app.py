from plotly.subplots import make_subplots
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import io, zipfile
import re

st.set_page_config(page_title='SpectraKinetics v9.3', layout='wide')

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

st.title("SpectraKinetics v9.3 — Enhanced Analysis")

data = st.session_state.datasets
if not data:
    st.stop()

# ANALYSIS
rows=[]
spectra_matrix=[]
labels=[]

for i,(name,d) in enumerate(data.items()):
    if 280 not in d['spectra']: continue

    wl = d['wavelengths']
    y = d['spectra'][280]

    i280,i340,i350,i330 = [nearest(wl,x) for x in (280,340,350,330)]

    irif = y[i280]/y[i340] if y[i340]!=0 else np.nan
    pie = y[i350]/y[i330] if y[i330]!=0 else np.nan

    peak_wl = wl[np.argmax(y)]
    peak_intensity = np.max(y)

    rows.append({
        "File":name,
        "Index":i,
        "IR/IF":irif,
        "I350/I330":pie,
        "λmax":peak_wl,
        "Peak Intensity":peak_intensity
    })

    spectra_matrix.append(y)
    labels.append(name)


df = pd.DataFrame(rows)
st.dataframe(df, use_container_width=True)

# -------- AVG + STD SPECTRA --------
st.header("Average Spectrum ± STD")

spectra_matrix = np.array(spectra_matrix)
avg = np.mean(spectra_matrix, axis=0)
stdev = np.std(spectra_matrix, axis=0)

fig_avg = go.Figure()
fig_avg.add_trace(go.Scatter(x=wl, y=avg, name='Mean'))
fig_avg.add_trace(go.Scatter(x=wl, y=avg+stdev, name='+STD', line=dict(dash='dot')))
fig_avg.add_trace(go.Scatter(x=wl, y=avg-stdev, name='-STD', line=dict(dash='dot')))

st.plotly_chart(fig_avg, use_container_width=True)

# -------- HEATMAP --------
st.header("Metric Heatmap")
heat = df[["IR/IF","I350/I330","λmax","Peak Intensity"]].values
fig_heat = px.imshow(heat, labels=dict(x="Metric", y="Sample", color="Value"))
st.plotly_chart(fig_heat, use_container_width=True)

# -------- DERIVATIVE SPECTRA --------
st.header("Derivative Spectra (280 nm)")
fig_deriv = go.Figure()
for name,d in data.items():
    if 280 in d['spectra']:
        deriv = np.gradient(d['spectra'][280])
        fig_deriv.add_trace(go.Scatter(x=d['wavelengths'], y=deriv, name=name))
st.plotly_chart(fig_deriv, use_container_width=True)

# -------- STANDARD SPECTRA --------
st.header("Spectra Overlay (280)")
fig280 = go.Figure()
for name,d in data.items():
    if 280 in d['spectra']:
        fig280.add_trace(go.Scatter(x=d['wavelengths'], y=d['spectra'][280], name=name))
st.plotly_chart(fig280, use_container_width=True)

# EXPORT
st.header("Export")

def build_zip():
    buf=io.BytesIO()
    with zipfile.ZipFile(buf,'w') as z:
        z.writestr('analysis.csv', df.to_csv(index=False))
    buf.seek(0)
    return buf

if st.button('Build ZIP'):
    st.download_button('Download ZIP', build_zip(), 'spectrakinetics_v9_3.zip')
