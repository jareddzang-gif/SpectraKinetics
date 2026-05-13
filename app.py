import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import io, zipfile

st.set_page_config(page_title='SpectraKinetics Batch', layout='wide')

if 'datasets' not in st.session_state:
    st.session_state.datasets = {}

def parse_processed_file(file_bytes, filename):
    content = file_bytes.decode('utf-8', errors='replace').splitlines()
    spectra = {}
    wavelengths = []
    ex_vals = []

    for i, line in enumerate(content):
        parts = line.split('	')
        if len(parts) > 0 and 'excitation wavelength' in parts[0].lower():
            for val in parts[2:]:
                try:
                    ex_vals.append(float(val))
                except:
                    pass
        if len(parts) > 0 and parts[0].isdigit():
            data_start = i
            break

    matrix = []
    for line in content[data_start:]:
        parts = line.split('	')
        if len(parts) < 3:
            continue
        try:
            wl = float(parts[1])
            intensities = [float(x) for x in parts[2:2+len(ex_vals)]]
            wavelengths.append(wl)
            matrix.append(intensities)
        except:
            continue

    matrix = np.array(matrix)
    for j, ex in enumerate(ex_vals):
        spectra[ex] = matrix[:, j]

    return {'wavelengths': np.array(wavelengths), 'spectra': spectra, 'filename': filename}, None


def nearest(arr, val):
    arr = np.array(arr)
    idx = np.argmin(np.abs(arr - val))
    return arr[idx], idx


with st.sidebar:
    uploaded = st.file_uploader('Upload up to 200 files', type=['txt'], accept_multiple_files=True)

    if uploaded:
        if len(uploaded) > 200:
            uploaded = uploaded[:200]
        for f in uploaded:
            if f.name not in st.session_state.datasets:
                data, _ = parse_processed_file(f.read(), f.name)
                st.session_state.datasets[f.name] = data

    ex_choice = st.selectbox('Excitation', [280,260])

st.title('SpectraKinetics Batch')

datasets = st.session_state.datasets
if not datasets:
    st.stop()

# Spectra plot
fig = go.Figure()
for fname, d in datasets.items():
    if ex_choice in d['spectra']:
        fig.add_trace(go.Scatter(x=d['wavelengths'], y=d['spectra'][ex_choice], name=fname))

st.plotly_chart(fig, use_container_width=True)

# Analysis
results=[]
for fname, d in datasets.items():
    wl = d['wavelengths']
    if 280 not in d['spectra']:
        continue
    spec = d['spectra'][280]

    _,i280=nearest(wl,280)
    _,i340=nearest(wl,340)
    _,i350=nearest(wl,350)
    _,i330=nearest(wl,330)

    IR=spec[i280]
    IF=spec[i340]
    ratio_irif = IR/IF if IF!=0 else np.nan

    ratio_pie = spec[i350]/spec[i330] if spec[i330]!=0 else np.nan

    results.append({'File':fname,'IR/IF':ratio_irif,'I350/I330':ratio_pie})


df = pd.DataFrame(results)
st.dataframe(df, use_container_width=True)

fig2 = px.bar(df,x='File',y='IR/IF')
st.plotly_chart(fig2,use_container_width=True)

# EXPORT
st.header('Export Results')

def build_zip():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf,'w') as z:
        z.writestr('analysis.csv', df.to_csv(index=False))
        for fname,d in datasets.items():
            spec_df = pd.DataFrame({'Wavelength':d['wavelengths']})
            for ex,val in d['spectra'].items():
                spec_df[f'Ex_{ex}'] = val
            z.writestr(f'{fname}_spectra.csv', spec_df.to_csv(index=False))
    buf.seek(0)
    return buf

if st.button('Build ZIP Export'):
    zip_bytes = build_zip()
    st.download_button('Download ZIP', zip_bytes, 'spectrakinetics_export.zip','application/zip')
