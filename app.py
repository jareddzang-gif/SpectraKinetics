import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# CONFIG
st.set_page_config(page_title='SpectraKinetics Batch', layout='wide')

# SESSION STATE
if 'datasets' not in st.session_state:
    st.session_state.datasets = {}

# PARSER
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

    return {
        'wavelengths': np.array(wavelengths),
        'spectra': spectra,
        'filename': filename
    }, None


def nearest(arr, val):
    arr = np.array(arr)
    idx = np.argmin(np.abs(arr - val))
    return arr[idx], idx


# SIDEBAR
with st.sidebar:
    st.header('📁 Upload Files')

    uploaded = st.file_uploader(
        'Upload up to 200 files',
        type=['txt'],
        accept_multiple_files=True
    )

    if uploaded:
        if len(uploaded) > 200:
            st.warning('⚠️ Max 200 files allowed')
            uploaded = uploaded[:200]

        for f in uploaded:
            if f.name not in st.session_state.datasets:
                data, err = parse_processed_file(f.read(), f.name)
                if data:
                    st.session_state.datasets[f.name] = data
                else:
                    st.error(f"{f.name}: {err}")

    st.divider()

    ex_choice = st.selectbox('Excitation wavelength', [280, 260])


# MAIN
st.title('🔬 SpectraKinetics — Batch Mode')

if not st.session_state.datasets:
    st.info('Upload files to begin')
    st.stop()

datasets = st.session_state.datasets

# SPECTRA
st.header('🌈 Spectra Overlay')

fig = go.Figure()
colors = px.colors.qualitative.Vivid

for i, (fname, d) in enumerate(datasets.items()):
    if ex_choice not in d['spectra']:
        continue

    fig.add_trace(go.Scatter(
        x=d['wavelengths'],
        y=d['spectra'][ex_choice],
        mode='lines',
        name=fname,
        line=dict(color=colors[i % len(colors)])
    ))

fig.update_layout(
    xaxis_title='Emission Wavelength (nm)',
    yaxis_title='Intensity',
    template='plotly_white',
    title=f'Overlay Spectra (Ex {ex_choice} nm)'
)

st.plotly_chart(fig, use_container_width=True)


# ANALYSIS
st.header('🧪 Batch Analysis')

results = []

for fname, d in datasets.items():
    wl = d['wavelengths']

    if 280 not in d['spectra']:
        continue

    spec280 = d['spectra'][280]

    _, i280 = nearest(wl, 280)
    _, i340 = nearest(wl, 340)
    _, i350 = nearest(wl, 350)
    _, i330 = nearest(wl, 330)

    IR = spec280[i280]
    IF = spec280[i340]
    ratio_irif = IR / IF if IF != 0 else np.nan

    I350 = spec280[i350]
    I330 = spec280[i330]
    ratio_pie = I350 / I330 if I330 != 0 else np.nan

    val260 = np.nan
    if 260 in d['spectra']:
        spec260 = d['spectra'][260]
        val260 = spec260[i350]

    results.append({
        'File': fname,
        'IR/IF': ratio_irif,
        'I350/I330': ratio_pie,
        'I350 (Ex280)': I350,
        'I350 (Ex260)': val260
    })

=df = pd.DataFrame(results)

st.dataframe(df, use_container_width=True)

metric = st.selectbox('Select metric', ['IR/IF', 'I350/I330'])

fig2 = px.bar(df, x='File', y=metric, title=f'{metric} across files')

st.plotly_chart(fig2, use_container_width=True)


# RAW PREVIEW
st.header('🗃 Raw Preview')

sample_file = list(datasets.values())[0]

raw_df = pd.DataFrame({
    'Wavelength': sample_file['wavelengths'],
    'Intensity': sample_file['spectra'][list(sample_file['spectra'].keys())[0]]
})

st.dataframe(raw_df, use_container_width=True)
