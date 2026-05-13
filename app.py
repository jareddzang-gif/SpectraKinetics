import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import io, zipfile, re

st.set_page_config(page_title='SpectraKinetics v11.5 (Visual Upgrade)', layout='wide')

# NAV
page = st.sidebar.radio("Navigation", ["Spectra Analysis", "Kinetics"])

# CLEAN NAME

def clean_filename(name):
    match = re.search(r"(\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})", name)
    return match.group(1) if match else name[:15]

# PARSER (unchanged logically)
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

# ---------- VISUAL STYLE ----------
def styled_layout(fig, title):
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=20)),
        template='plotly_white',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
        margin=dict(l=40, r=40, t=70, b=40)
    )
    return fig

# =====================================================
if page == "Spectra Analysis":

    st.title("Spectra Analysis")

    rows=[]
    for i,(name,d) in enumerate(data.items()):
        if 280 not in d['spectra']: continue
        wl=d['wavelengths']; y=d['spectra'][280]

        ir_idx=np.argmax(y); ir_peak=wl[ir_idx]; ir_int=y[ir_idx]
        mask=(wl>=300)&(wl<=390)
        if np.any(mask):
            y_if=y[mask]; wl_if=wl[mask]; idx=np.argmax(y_if)
            if_peak=wl_if[idx]; if_int=y_if[idx]
        else:
            if_peak=np.nan; if_int=np.nan

        nearest=lambda v: np.argmin(np.abs(wl-v))
        irif=y[nearest(280)]/y[nearest(340)] if y[nearest(340)]!=0 else np.nan
        pie=y[nearest(350)]/y[nearest(330)] if y[nearest(330)]!=0 else np.nan

        rows.append({
            "File":name,"Index":i,"IR/IF":irif,"I350/I330":pie,
            "Aggregation Index":np.nan,"Concentration (mg/mL)":np.nan,
            "IR (nm)":ir_peak,"IR Peak Intensity":ir_int,
            "IF (nm)":if_peak,"IF Peak Intensity":if_int
        })

    df=pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True)

    # Spectra plot (styled only)
    fig=go.Figure()
    for name,d in data.items():
        if ex_toggle in d['spectra']:
            fig.add_trace(go.Scatter(x=d['wavelengths'], y=d['spectra'][ex_toggle], name=name, line=dict(width=2)))
    fig = styled_layout(fig, f"Spectra Overlay (Ex {ex_toggle} nm)")
    fig.update_xaxes(title="Wavelength (nm)")
    fig.update_yaxes(title="Intensity")
    st.plotly_chart(fig, use_container_width=True, key=f"spectra_{ex_toggle}")

    # APIES plot (styled)
    fig2=go.Figure()
    fig2.add_trace(go.Scatter(x=df['Index'], y=df['IR/IF'], name='IR/IF', mode='lines+markers', line=dict(width=3)))
    fig2.add_trace(go.Scatter(x=df['Index'], y=df['I350/I330'], name='I350/I330', mode='lines+markers', line=dict(width=3)))

    fig2 = styled_layout(fig2, "APIES (All Metrics Overlay)")
    fig2.update_xaxes(title="Sample Index")
    fig2.update_yaxes(title="Fluorescence Ratios")

    st.plotly_chart(fig2, use_container_width=True, key="apies_plot")

# =====================================================
if page == "Kinetics":

    st.title("Kinetics Analysis (Merged Timeline)")

    segments_280=[]; segments_350=[]
    sorted_items=sorted(data.items(), key=lambda x:x[0])
    time_offset=0

    for name,d in sorted_items:
        if d['kinetics'] is None: continue
        kin=d['kinetics']; times=kin['times']
        if len(times)==0: continue
        wl=kin['wavelengths']; matrix=kin['matrix']
        if len(wl)==0 or matrix.size==0: continue

        i280=np.argmin(np.abs(wl-280)); i350=np.argmin(np.abs(wl-350))
        s280=matrix[i280,:]; s350=matrix[i350,:]

        t=times+time_offset
        segments_280.append((t,s280)); segments_350.append((t,s350))
        time_offset += (times[-1]-times[0])

    if segments_280:
        fig_k=go.Figure()
        for t,y in segments_280:
            fig_k.add_trace(go.Scatter(x=t,y=y,name='280 nm (IR)',line=dict(color='#1f77b4',width=2)))
        for t,y in segments_350:
            fig_k.add_trace(go.Scatter(x=t,y=y,name='350 nm (IF)',line=dict(color='#d62728',width=2)))

        fig_k = styled_layout(fig_k, "Merged Kinetics Timeline")
        fig_k.update_xaxes(title="Time (s)")
        fig_k.update_yaxes(title="Intensity")

        st.plotly_chart(fig_k, use_container_width=True, key="kinetics_merged")
    else:
        st.info("No valid kinetics data detected.")

# EXPORT
st.sidebar.markdown("---")
if st.sidebar.button("Download Analysis CSV"):
    buf=io.BytesIO()
    with zipfile.ZipFile(buf,'w') as z:
        z.writestr('analysis.csv', df.to_csv(index=False) if 'df' in locals() else "")
    buf.seek(0)
    st.sidebar.download_button("Download", buf, "spectrakinetics_v11_5_visual.zip")
