import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import io, zipfile
from scipy.integrate import simpson
from scipy.stats import linregress

st.set_page_config(page_title='SpectraKinetics v8', layout='wide')

if 'datasets' not in st.session_state:
    st.session_state.datasets = {}

# PARSER

def parse_file(file_bytes, filename):
    content = file_bytes.decode('utf-8', errors='replace').splitlines()
    spectra, wavelengths, ex_vals = {}, [], []

    for i, line in enumerate(content):
        parts = line.split('	')
        if 'excitation wavelength' in parts[0].lower():
            ex_vals = [float(v) for v in parts[2:] if v]
        if parts[0].isdigit():
            data_start = i
            break

    matrix = []
    for line in content[data_start:]:
        parts = line.split('	')
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
    idx = np.argmin(np.abs(arr - val))
    return idx

# SIDEBAR
with st.sidebar:
    files = st.file_uploader("Upload ≤200 files", type=['txt'], accept_multiple_files=True)

    if files:
        st.session_state.datasets = {}
        for f in files[:200]:
            st.session_state.datasets[f.name] = parse_file(f.read(), f.name)

    ex_choice = st.selectbox('Excitation', [280, 260])

st.title("SpectraKinetics v8 — Full Pipeline")

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
blue_shift_values=[]
file_index=[]

for i,(name,d) in enumerate(data.items()):
    if 280 not in d['spectra']: continue

    wl = d['wavelengths']
    y = d['spectra'][280]

    i280,i340,i350,i330 = [nearest(wl,x) for x in (280,340,350,330)]

    irif = y[i280]/y[i340] if y[i340]!=0 else np.nan
    pie = y[i350]/y[i330] if y[i330]!=0 else np.nan

    # area under curve (Rayleigh ~260-300, Fluorescence ~300-400)
    mask_ray = (wl>=260)&(wl<=300)
    mask_flu = (wl>=300)&(wl<=400)

    auc_ray = simpson(y[mask_ray], wl[mask_ray])
    auc_flu = simpson(y[mask_flu], wl[mask_flu])

    shift_ratio = auc_ray/auc_flu if auc_flu!=0 else np.nan

    rows.append({"File":name,"Index":i,"IR/IF":irif,"I350/I330":pie,"Shift Ratio":shift_ratio})

    blue_shift_values.append(shift_ratio)
    file_index.append(i)


df = pd.DataFrame(rows)
st.dataframe(df, use_container_width=True)

# LINE GRAPH IR/IF
fig_line = px.line(df.sort_values("Index"), x="Index", y="IR/IF", markers=True, title="IR/IF Trend")
st.plotly_chart(fig_line, use_container_width=True)

# LINEAR REGRESSION (blue shift)
if len(blue_shift_values) > 1:
    slope, intercept, r, p, _ = linregress(file_index, blue_shift_values)
    reg_line = [intercept + slope*x for x in file_index]

    fig_reg = go.Figure()
    fig_reg.add_trace(go.Scatter(x=file_index,y=blue_shift_values, mode='markers', name='Shift'))
    fig_reg.add_trace(go.Scatter(x=file_index,y=reg_line, mode='lines', name=f'Fit (R²={r**2:.3f})'))
    fig_reg.update_layout(title="Blue Shift Regression")
    st.plotly_chart(fig_reg, use_container_width=True)

# EXPORT
st.header("Export Full Bundle")

def build_zip():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf,'w') as z:
        z.writestr("analysis.csv", df.to_csv(index=False))

        for name,d in data.items():
            spec = pd.DataFrame({'Wavelength':d['wavelengths']})
            for ex,val in d['spectra'].items():
                spec[f'Ex_{ex}'] = val
            z.writestr(f"{name}_spectra.csv", spec.to_csv(index=False))

    buf.seek(0)
    return buf

if st.button("Build ZIP"):
    st.download_button("Download", build_zip(), "spectrakinetics_v8.zip")
