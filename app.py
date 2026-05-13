import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import io, zipfile

st.set_page_config(page_title='SpectraKinetics v8', layout='wide')

if 'datasets' not in st.session_state:
    st.session_state.datasets = {}

# ✅ UNIVERSAL AUC (NO NUMPY/SCIPY DEPENDENCY)
def safe_trapz(y, x):
    y = np.array(y)
    x = np.array(x)
    return np.sum((y[1:] + y[:-1]) * (x[1:] - x[:-1]) / 2)

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
    return np.argmin(np.abs(arr - val))

# SIDEBAR
with st.sidebar:
    files = st.file_uploader("Upload ≤200 files", type=['txt'], accept_multiple_files=True)

    if files:
        st.session_state.datasets = {}
        for f in files[:200]:
            st.session_state.datasets[f.name] = parse_file(f.read(), f.name)

    ex_choice = st.selectbox('Excitation', [280, 260])

st.title("SpectraKinetics v8 — Cloud-Safe Final")

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

    mask_ray = (wl>=260)&(wl<=300)
    mask_flu = (wl>=300)&(wl<=400)

    # ✅ FINAL FIX: custom trapezoid
    auc_ray = safe_trapz(y[mask_ray], wl[mask_ray])
    auc_flu = safe_trapz(y[mask_flu], wl[mask_flu])

    shift_ratio = auc_ray/auc_flu if auc_flu!=0 else np.nan

    rows.append({"File":name,"Index":i,"IR/IF":irif,"I350/I330":pie,"Shift Ratio":shift_ratio})

    blue_shift_values.append(shift_ratio)
    file_index.append(i)


df = pd.DataFrame(rows)
st.dataframe(df, use_container_width=True)

# IR/IF Line Plot
fig_line = px.line(df.sort_values("Index"), x="Index", y="IR/IF", markers=True, title="IR/IF Trend")
st.plotly_chart(fig_line, use_container_width=True)

# Regression
if len(blue_shift_values) > 1:
    x = np.array(file_index)
    y_vals = np.array(blue_shift_values)

    coeffs = np.polyfit(x, y_vals, 1)
    reg_line = coeffs[0]*x + coeffs[1]

    ss_res = np.sum((y_vals - reg_line)**2)
    ss_tot = np.sum((y_vals - np.mean(y_vals))**2)
    r2 = 1 - (ss_res/ss_tot if ss_tot!=0 else 0)

    fig_reg = go.Figure()
    fig_reg.add_trace(go.Scatter(x=x,y=y_vals, mode='markers', name='Shift'))
    fig_reg.add_trace(go.Scatter(x=x,y=reg_line, mode='lines', name=f'Fit (R²={r2:.3f})'))
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
    st.download_button("Download", build_zip(), "spectrakinetics_v8_final.zip")
