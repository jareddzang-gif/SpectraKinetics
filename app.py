from plotly.subplots import make_subplots
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import io, zipfile

st.set_page_config(page_title='SpectraKinetics v9.2', layout='wide')

if 'datasets' not in st.session_state:
    st.session_state.datasets = {}

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

    return {'wavelengths': np.array(wavelengths), 'spectra': spectra, 'filename': filename}


def nearest(arr, val):
    return np.argmin(np.abs(arr - val))

# SIDEBAR
with st.sidebar:
    files = st.file_uploader("Upload ≤200 files", type=['txt'], accept_multiple_files=True)

    if files:
        st.session_state.datasets = {}
        for f in files[:200]:
            st.session_state.datasets[f.name] = parse_file(f.read(), f.name)

st.title("SpectraKinetics v9.2 — Dual Axis Clean View")

data = st.session_state.datasets
if not data:
    st.stop()

# ANALYSIS
rows=[]

for i,(name,d) in enumerate(data.items()):
    if 280 not in d['spectra']: continue

    wl = d['wavelengths']
    y = d['spectra'][280]

    i280,i340,i350,i330 = [nearest(wl,x) for x in (280,340,350,330)]

    irif = y[i280]/y[i340] if y[i340]!=0 else np.nan
    pie = y[i350]/y[i330] if y[i330]!=0 else np.nan

    peak_wl = wl[np.argmax(y)]

    concentration = np.nan
    agg_index = np.nan

    rows.append({
        "File":name,
        "Index":i,
        "IR/IF":irif,
        "I350/I330":pie,
        "λmax":peak_wl,
        "Concentration":concentration,
        "Aggregation Index":agg_index
    })


df = pd.DataFrame(rows)
st.dataframe(df, use_container_width=True)

# SPECTRA
st.header("Spectra Overlay (Ex 280)")
fig280 = go.Figure()
for name,d in data.items():
    if 280 in d['spectra']:
        fig280.add_trace(go.Scatter(x=d['wavelengths'], y=d['spectra'][280], name=name))
st.plotly_chart(fig280, use_container_width=True)

st.header("Spectra Overlay (Ex 260)")
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

# NORMALIZATION FOR CLARITY
norm_df = df.copy()
for col in ['IR/IF','I350/I330']:
    if norm_df[col].notna().any():
        norm_df[col] = (norm_df[col] - norm_df[col].min())/(norm_df[col].max()-norm_df[col].min()+1e-9)

# COMBINED PLOT WITH DUAL AXIS
st.header("Normalized Metrics + Absorbance (Dual Axis)")

fig = go.Figure()

# primary axis (normalized metrics)
fig.add_trace(go.Scatter(x=norm_df['Index'], y=norm_df['IR/IF'], mode='lines+markers', name='IR/IF (norm)', line=dict(width=3)))
fig.add_trace(go.Scatter(x=norm_df['Index'], y=norm_df['I350/I330'], mode='lines+markers', name='I350/I330 (norm)', line=dict(width=3)))

# secondary axis
if df['Concentration'].notna().any():
    fig.add_trace(go.Scatter(x=df['Index'], y=df['Concentration'], mode='lines+markers', name='Concentration', yaxis='y2'))

if df['Aggregation Index'].notna().any():
    fig.add_trace(go.Scatter(x=df['Index'], y=df['Aggregation Index'], mode='lines+markers', name='Aggregation Index', yaxis='y2'))

fig.update_layout(
    yaxis=dict(title="Normalized Metrics"),
    yaxis2=dict(title="Absorbance Metrics", overlaying='y', side='right'),
    xaxis_title="Sample Index",
    title="Overlay with Dual Axis"
)

st.plotly_chart(fig, use_container_width=True)

# EXPORT
st.header("Export")

def build_zip():
    buf=io.BytesIO()
    with zipfile.ZipFile(buf,'w') as z:
        z.writestr('analysis.csv', df.to_csv(index=False))
    buf.seek(0)
    return buf

if st.button('Build ZIP'):
    st.download_button('Download ZIP', build_zip(), 'spectrakinetics_v9_2.zip')
