from plotly.subplots import make_subplots
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import io, zipfile

st.set_page_config(page_title='SpectraKinetics v9', layout='wide')

if 'datasets' not in st.session_state:
    st.session_state.datasets = {}

# SAFE AUC

def safe_trapz(y, x):
    y = np.array(y)
    x = np.array(x)
    return np.sum((y[1:] + y[:-1]) * (x[1:] - x[:-1]) / 2)

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

st.title("SpectraKinetics v9 — Advanced Analysis")

data = st.session_state.datasets
if not data:
    st.stop()

# SPECTRA
st.header("Spectra Overlay")
fig = go.Figure()
for name,d in data.items():
    if ex_choice in d['spectra']:
        fig.add_trace(go.Scatter(x=d['wavelengths'], y=d['spectra'][ex_choice], name=name))
st.plotly_chart(fig, use_container_width=True)

# ANALYSIS
st.header("Batch Analysis")
rows=[]

for i,(name,d) in enumerate(data.items()):
    if 280 not in d['spectra']: continue

    wl = d['wavelengths']
    y = d['spectra'][280]

    i280,i340,i350,i330 = [nearest(wl,x) for x in (280,340,350,330)]

    irif = y[i280]/y[i340] if y[i340]!=0 else np.nan
    pie = y[i350]/y[i330] if y[i330]!=0 else np.nan

    peak_wl = wl[np.argmax(y)]

    rows.append({
        "File":name,
        "Index":i,
        "IR/IF":irif,
        "I350/I330":pie,
        "Peak λmax":peak_wl
    })


df = pd.DataFrame(rows)
st.dataframe(df, use_container_width=True)

# MULTI-PANEL FIGURE
st.header("Multi-panel Figures")

fig_multi = make_subplots(rows=2, cols=2,
    subplot_titles=("IR/IF Trend","I350/I330 Trend","λmax Trend","Spectra (280 nm)"))

# IR/IF
fig_multi.add_trace(go.Scatter(x=df['Index'], y=df['IR/IF'], mode='lines+markers'), row=1, col=1)

# I350/I330
fig_multi.add_trace(go.Scatter(x=df['Index'], y=df['I350/I330'], mode='lines+markers'), row=1, col=2)

# lambda max
fig_multi.add_trace(go.Scatter(x=df['Index'], y=df['Peak λmax'], mode='lines+markers'), row=2, col=1)

# spectra overlay
for name,d in data.items():
    if 280 in d['spectra']:
        fig_multi.add_trace(go.Scatter(x=d['wavelengths'], y=d['spectra'][280], name=name), row=2, col=2)

st.plotly_chart(fig_multi, use_container_width=True)

# CORRELATION
st.header("λmax vs IR/IF Correlation")
fig_corr = px.scatter(df, x="Peak λmax", y="IR/IF", trendline="ols")
st.plotly_chart(fig_corr, use_container_width=True)

# PCA
st.header("PCA Clustering")

matrix = []
labels = []
for name,d in data.items():
    if 280 in d['spectra']:
        matrix.append(d['spectra'][280])
        labels.append(name)

matrix = np.array(matrix)

# PCA via SVD
matrix_centered = matrix - matrix.mean(axis=0)
U,S,Vt = np.linalg.svd(matrix_centered, full_matrices=False)
coords = U[:, :2] @ np.diag(S[:2])

pca_df = pd.DataFrame(coords, columns=['PC1','PC2'])
pca_df['File'] = labels

fig_pca = px.scatter(pca_df, x='PC1', y='PC2', text='File')
st.plotly_chart(fig_pca, use_container_width=True)

# EXPORT
st.header("Export Report")

def build_zip():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf,'w') as z:
        z.writestr("analysis.csv", df.to_csv(index=False))
        z.writestr("pca.csv", pca_df.to_csv(index=False))

        for name,d in data.items():
            spec = pd.DataFrame({'Wavelength':d['wavelengths']})
            for ex,val in d['spectra'].items():
                spec[f'Ex_{ex}'] = val
            z.writestr(f"{name}_spectra.csv", spec.to_csv(index=False))

    buf.seek(0)
    return buf

if st.button("Build ZIP"):
    st.download_button("Download", build_zip(), "spectrakinetics_v9.zip")

