from plotly.subplots import make_subplots
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import io, zipfile

st.set_page_config(page_title='SpectraKinetics v9.1', layout='wide')

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

    ex_choice = st.selectbox('Excitation', [280, 260])

st.title("SpectraKinetics v9.1 — Clean Metrics View")

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

    # optional absorbance-based placeholders
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

# COMBINED MULTI-METRIC PLOT
st.header("Combined Metrics Overlay")

fig = go.Figure()

fig.add_trace(go.Scatter(x=df['Index'], y=df['IR/IF'], mode='lines+markers', name='IR/IF'))
fig.add_trace(go.Scatter(x=df['Index'], y=df['I350/I330'], mode='lines+markers', name='I350/I330'))

# only plot optional metrics if data exists
if df['Concentration'].notna().any():
    fig.add_trace(go.Scatter(x=df['Index'], y=df['Concentration'], mode='lines+markers', name='Concentration'))

if df['Aggregation Index'].notna().any():
    fig.add_trace(go.Scatter(x=df['Index'], y=df['Aggregation Index'], mode='lines+markers', name='Aggregation Index'))

fig.update_layout(
    title="All Metrics Overlay",
    xaxis_title="Sample Index",
    yaxis_title="Value"
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
    st.download_button('Download ZIP', build_zip(), 'spectrakinetics_v9_1_clean.zip')
