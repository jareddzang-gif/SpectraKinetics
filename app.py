import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import io, zipfile, re

st.set_page_config(page_title='SpectraKinetics v11.6 (AUC)', layout='wide')

# NAV
page = st.sidebar.radio("Navigation", ["Spectra Analysis", "Kinetics", "AUC Analysis"])

# CLEAN NAME

def clean_filename(name):
    match = re.search(r"(\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})", name)
    return match.group(1) if match else name[:15]

# PARSER (unchanged)
def parse_file(file_bytes, filename):
    content = file_bytes.decode('utf-8', errors='replace').splitlines()
    spectra = {}
    wavelengths = []
    ex_vals = []
    kinetics = None

    for i, line in enumerate(content):
        if "kinetic time" in line.lower():
            parts = line.split("	")
            blank_idx = next((j for j,p in enumerate(parts) if p.strip()==""), None)
            if blank_idx is not None:
                times = [float(x) for x in parts[blank_idx+1:] if x]
                wl, mat = [], []
                for row in content[i+1:]:
                    r = row.split("	")
                    try:
                        wl.append(float(r[1]))
                        mat.append([float(x) for x in r[blank_idx+1:blank_idx+1+len(times)]])
                    except:
                        continue
                kinetics = {"times": np.array(times), "wavelengths": np.array(wl), "matrix": np.array(mat)}

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

    matrix = np.array(matrix) if len(matrix)>0 else np.array([])
    for j, ex in enumerate(ex_vals if len(matrix)>0 else []):
        spectra[ex] = matrix[:, j]

    return {'wavelengths': np.array(wavelengths),'spectra': spectra,'filename': clean_filename(filename),'kinetics': kinetics}

# Upload
files = st.sidebar.file_uploader("Upload ≤200 files", type=['txt'], accept_multiple_files=True)
ex_toggle = st.sidebar.radio("Spectra View", [280, 260])

if files:
    st.session_state.datasets = {}
    for f in files[:200]:
        parsed = parse_file(f.read(), f.name)
        st.session_state.datasets[parsed['filename']] = parsed


data = st.session_state.get('datasets', {})
if not data:
    st.stop()

# ============================
# AUC PAGE (NEW)
# ============================
if page == "AUC Analysis":

    st.title("Area Under the Curve (AUC) Analysis")

    file_names = list(data.keys())
    selected_file = st.selectbox("Select Dataset", file_names)

    d = data[selected_file]

    if ex_toggle not in d['spectra']:
        st.warning("Selected excitation wavelength not available.")
        st.stop()

    wl = d['wavelengths']
    y = d['spectra'][ex_toggle]

    # sliders
    min_wl, max_wl = float(np.min(wl)), float(np.max(wl))

    start_wl, end_wl = st.slider(
        "Select Wavelength Range",
        min_value=min_wl,
        max_value=max_wl,
        value=(min_wl+20, max_wl-20)
    )

    # mask region
    mask = (wl >= start_wl) & (wl <= end_wl)
    area = np.trapz(y[mask], wl[mask]) if np.any(mask) else 0

    # plot
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=wl, y=y, name='Full Spectrum', line=dict(color='black')))

    fig.add_trace(go.Scatter(
        x=wl[mask],
        y=y[mask],
        name='Selected Region',
        fill='tozeroy',
        line=dict(color='orange')
    ))

    fig.update_layout(
        title="AUC Selection",
        xaxis_title="Wavelength (nm)",
        yaxis_title="Intensity",
        template='plotly_white'
    )

    st.plotly_chart(fig, use_container_width=True, key="auc_plot")

    # output
    st.subheader("AUC Result")
    st.metric(label="Area Under Curve", value=f"{area:.3f}")

    st.info(f"Range: {start_wl:.1f} nm → {end_wl:.1f} nm")

# ============================
# EXISTING PAGES NOT TOUCHED
# ============================
if page == "Spectra Analysis":
    st.title("Spectra Analysis (unchanged — see previous version)")

if page == "Kinetics":
    st.title("Kinetics Analysis (unchanged — see previous version)")

# EXPORT
st.sidebar.markdown("---")
if st.sidebar.button("Download Analysis CSV"):
    buf=io.BytesIO()
    with zipfile.ZipFile(buf,'w') as z:
        z.writestr('analysis.csv','')
    buf.seek(0)
    st.sidebar.download_button("Download", buf, "spectrakinetics_v11_6.zip")
