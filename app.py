# v17 FINAL STABLE (APIES + REGRESSION + CLEAN AUC)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import re

st.set_page_config(page_title='SpectraKinetics v17', layout='wide')

page = st.sidebar.radio("Navigation",
                        ["APIES Dashboard", "AUC Analysis"])

ex_toggle = st.sidebar.radio("Excitation Wavelength (nm)", [280, 260])

# =====================
# ✅ TIMESTAMP EXTRACTION (YOUR FORMAT)
# =====================
def extract_time(name):
    match = re.search(r"(\d{4})_(\d{2})_(\d{2})_(\d{2})_(\d{2})_(\d{2})", name)
    if match:
        return pd.to_datetime(
            f"{match.group(1)}-{match.group(2)}-{match.group(3)} "
            f"{match.group(4)}:{match.group(5)}:{match.group(6)}"
        )
    return pd.NaT

# =====================
# PARSER
# =====================

def parse_file(file_bytes, filename):

    content = file_bytes.decode("utf-8", errors="replace").splitlines()
# ---------- CASE 0: ATEEM HEADER FORMAT ----------
    for i, line in enumerate(content):
        if "excitation wavelength" in line.lower():

            parts = re.split(r"\s+|\t+", line.strip())
            ex_vals = []

            for val in parts[2:]:
                try:
                    ex_vals.append(float(val))
                except:
                    continue

            wavelengths, matrix = [], []

            for line in content[i+1:]:
                parts = re.split(r"\s+|\t+", line.strip())

                if len(parts) < 3:
                    continue

                try:
                    wavelengths.append(float(parts[1]))
                    matrix.append([float(x) for x in parts[2:2+len(ex_vals)]])
                except:
                    continue

            matrix = np.array(matrix)

            spectra = {}
            for j, ex in enumerate(ex_vals):
                spectra[ex] = matrix[:, j]

            return {
                "wavelengths": np.array(wavelengths),
                "spectra": spectra,
                "filename": filename
            }

  

    # ---------- CASE 1: SIMPLE 2-COLUMN (IFEABS) ----------
    test = re.split(r"\s+|\t+", content[0].strip())
    if len(test) == 2:
        wavelengths, values = [], []

        for line in content:
            parts = re.split(r"\s+|\t+", line.strip())
            if len(parts) >= 2:
                try:
                    wavelengths.append(float(parts[0]))
                    values.append(float(parts[1]))
                except:
                    continue

        return {
            "wavelengths": np.array(wavelengths),
            "spectra": {0: np.array(values)},
            "filename": filename
        }

    # ---------- CASE 2: COLUMN HEADER FORMAT (IFEPEM) ----------
    header = re.split(r"\s+|\t+", content[0].strip())

    ex_vals = []
    for val in header[1:]:
        try:
            ex_vals.append(float(val))
        except:
            continue

    if len(ex_vals) > 0:
        wavelengths, matrix = [], []

        for line in content[2:]:  # skip header + unit row
            parts = re.split(r"\s+|\t+", line.strip())
            if len(parts) < len(ex_vals) + 1:
                continue
            try:
                wavelengths.append(float(parts[0]))
                matrix.append([float(x) for x in parts[1:1+len(ex_vals)]])
            except:
                continue

        matrix = np.array(matrix)

        spectra = {}
        for j, ex in enumerate(ex_vals):
            spectra[ex] = matrix[:, j]

        return {
            "wavelengths": np.array(wavelengths),
            "spectra": spectra,
            "filename": filename
        }

    # ---------- FALLBACK ----------
    return {
        "wavelengths": np.array([]),
        "spectra": {},
        "filename": filename
    }


def apply_ife_correction(pem, abs_data):

    wl_em = pem["wavelengths"]
    spectra = pem["spectra"]

    wl_abs = abs_data["wavelengths"]
    abs_vals = list(abs_data["spectra"].values())[0]

    corrected = {}

    for ex, y in spectra.items():

        A_em = np.interp(wl_em, wl_abs, abs_vals)
        
        if ex < wl_abs.min() or ex > wl_abs.max():
            corrected[ex] = y
            continue

        A_ex = np.interp(ex, wl_abs, abs_vals)

        factor = 10 ** ((A_ex + A_em) / 2)

        corrected[ex] = y * factor

    return corrected


# =====================
# LOAD FILES
# =====================

files = st.sidebar.file_uploader("Upload", type=["txt", "dat"], accept_multiple_files=True)

if files:
    st.session_state.datasets = {}

    for f in files:
        name = f.name
        parsed = parse_file(f.read(), name)

        if "IFEPEM" in name:
            key = re.sub(r"_EEM_IFE(P|A)BS|_EEM_IFEPEM", "", name)
            st.session_state.datasets.setdefault(key, {})["pem"] = parsed

        elif "IFEABS" in name:
            key = re.sub(r"_EEM_IFE(P|A)BS|_EEM_IFEPEM", "", name)
            st.session_state.datasets.setdefault(key, {})["abs"] = parsed

        else:
            # handle standalone datasets (ATEEM files)
            key = name
            st.session_state.datasets[key] = {"pem": parsed}

data_raw = st.session_state.get("datasets", {})

data = {}

for name, pair in data_raw.items():

    if "pem" not in pair:
        continue

    pem = pair["pem"]

    if "abs" in pair:
        spectra = apply_ife_correction(pem, pair["abs"])
    else:
        spectra = pem["spectra"]

    data[name] = {
        "wavelengths": pem["wavelengths"],
        "spectra": spectra,
        "filename": name
    }

if not data:
    st.info("Upload data to begin.")
    st.stop()


# =====================
# ✅ APIES DASHBOARD
# =====================
if page == "APIES Dashboard":

    st.title("APIES Dashboard")

    col1, col2 = st.columns(2)

    with col1:
        ir_start = st.number_input("IR Start", value=270.0)
        ir_end = st.number_input("IR End", value=300.0)

    with col2:
        if_start = st.number_input("IF Start", value=320.0)
        if_end = st.number_input("IF End", value=390.0)

    ir_start, ir_end = sorted([ir_start, ir_end])
    if_start, if_end = sorted([if_start, if_end])

    rows = []

    for i, (name, d) in enumerate(data.items()):

        if ex_toggle not in d["spectra"]:
            continue

        wl = d["wavelengths"]
        y = d["spectra"][ex_toggle]


        mask_ir = (wl >= ir_start) & (wl <= ir_end)
        mask_if = (wl >= if_start) & (wl <= if_end)
        
        auc_ir = np.trapezoid(y[mask_ir], wl[mask_ir]) if np.any(mask_ir) else np.nan
        auc_if = np.trapezoid(y[mask_if], wl[mask_if]) if np.any(mask_if) else np.nan

        
        irif_auc = auc_ir / auc_if if np.isfinite(auc_if) and auc_if != 0 else np.nan
        
        idx_330 = np.argmin(np.abs(wl - 330))
        idx_350 = np.argmin(np.abs(wl - 350))

        ratio = y[idx_350] / y[idx_330] if y[idx_330] != 0 else np.nan


        rows.append({
            "File": name,
            "Time": extract_time(name),
            "Index": i,
            "IR/IF (AUC)": irif_auc,
            "I350/I330": ratio,
            "AUC IR": auc_ir,
            "AUC IF": auc_if
        })

    
    df = pd.DataFrame(rows)
    df = df.sort_values(by="Time", na_position="last").reset_index(drop=True)

    st.dataframe(df, use_container_width=True)

    # ✅ Spectra overlay
    st.subheader("Spectra Overlay")
    fig_spec = go.Figure()
    for name, d in data.items():
        if ex_toggle in d["spectra"]:
            fig_spec.add_trace(go.Scatter(
                x=d["wavelengths"],
                y=d["spectra"][ex_toggle],
                name=name
            ))
    fig_spec.update_layout(
        title=f"Fluorescence Emission Spectra (Ex = {ex_toggle} nm)",
        xaxis_title="Emission Wavelength (nm)",
        yaxis_title="Fluorescence Intensity (Counts/µA)",
        template="plotly_white"
    )
    
    st.plotly_chart(fig_spec, use_container_width=True)

    # =====================
    # ✅ APIES MULTI-AXIS + REGRESSION
    # =====================
    st.subheader("APIES Metrics with Regression")

    x_vals = df["Time"].fillna(df["Index"])

    fig = go.Figure()

    y_irif = df["IR/IF (AUC)"].values
      
    x_num_full = np.arange(len(df))
    valid = np.isfinite(y_irif)

    if np.sum(valid) > 1:
        x_num = x_num_full[valid]
        y_clean = y_irif[valid]

        coeffs = np.polyfit(x_num, y_clean, 1)

        fit = np.full_like(y_irif, np.nan, dtype=float)
        fit[valid] = np.polyval(coeffs, x_num)

        ss_res = np.sum((y_clean - np.polyval(coeffs, x_num))**2)
        ss_tot = np.sum((y_clean - np.mean(y_clean))**2)

        r2 = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan
    else:
        fit = y_irif
        r2 = np.nan


    # ✅ primary axis
    fig.add_trace(go.Scatter(
        x=x_vals,
        y=y_irif,
        name=f"IR/IF (AUC) R²={r2:.3f}",
        mode="lines+markers",
        yaxis="y1"
    ))

    fig.add_trace(go.Scatter(
        x=x_vals,
        y=fit,
        name="Fit",
        line=dict(dash="dash"),
        yaxis="y1"
    ))

    # ✅ secondary axis
    fig.add_trace(go.Scatter(
        x=x_vals,
        y=df["I350/I330"],
        name="I350/I330",
        mode="lines+markers",
        yaxis="y2"
    ))

    fig.update_layout(
        xaxis_title="Time / Sample",
        yaxis=dict(title="IR/IF (AUC)", side="left"),
        yaxis2=dict(title="I350/I330", overlaying='y', side='right'),
        template="plotly_white"
    )

    st.plotly_chart(fig, use_container_width=True)

    # ✅ AUC COMPONENTS PANEL
    st.subheader("AUC Components")

    fig_auc = go.Figure()

    fig_auc.add_trace(go.Scatter(
        x=x_vals,
        y=df["AUC IR"],
        name="AUC IR",
        mode="lines+markers"
    ))

    fig_auc.add_trace(go.Scatter(
        x=x_vals,
        y=df["AUC IF"],
        name="AUC IF",
        mode="lines+markers"
    ))

    st.plotly_chart(fig_auc, use_container_width=True)

# =====================
# ✅ AUC ANALYSIS (FIXED)
# =====================
if page == "AUC Analysis":

    st.title("AUC Analysis")

    selected = st.selectbox("Dataset", list(data.keys()))
    d = data[selected]

    if ex_toggle not in d["spectra"]:
        st.warning("No excitation data")
        st.stop()

    wl = d["wavelengths"]
    y = d["spectra"][ex_toggle]

    start_wl = st.number_input("Start WL", value=float(min(wl)+20))
    end_wl = st.number_input("End WL", value=float(max(wl)-20))

    start_wl, end_wl = sorted([start_wl, end_wl])

    mask = (wl >= start_wl) & (wl <= end_wl)
    area = np.trapezoid(y[mask], wl[mask]) if np.any(mask) else 0

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=wl, y=y, name="Spectrum"))

    if np.any(mask):
        fig.add_trace(go.Scatter(
            x=wl[mask], y=y[mask],
            fill="tozeroy",
            name="AUC Region"
        ))

    st.plotly_chart(fig)
    st.metric("AUC", f"{area:.3f}")

    # ✅ BATCH
    if st.button("Calculate AUC for All Datasets"):

        results = []

        for name, dataset in data.items():

            if ex_toggle not in dataset["spectra"]:
                continue

            wl_f = dataset["wavelengths"]
            y_f = dataset["spectra"][ex_toggle]

            mask = (wl_f >= start_wl) & (wl_f <= end_wl)
            if not np.any(mask):
                continue

            auc_val = np.trapezoid(y_f[mask], wl_f[mask])

            t = extract_time(name)

            if pd.isna(t):
                continue

            results.append({"time": t, "AUC": auc_val})

        df_auc = pd.DataFrame(results)

        if df_auc.empty:
            st.warning("No valid time data")
            st.stop()

        df_auc = df_auc.groupby("time", as_index=False).mean().sort_values("time")

        st.dataframe(df_auc)

        x = np.arange(len(df_auc))
        y_vals = df_auc["AUC"].values

        
        valid = np.isfinite(y_vals)
  
        if np.sum(valid) > 1:
            x_clean = x[valid]
            y_clean = y_vals[valid]
            coeffs = np.polyfit(x_clean, y_clean, 1)

            fit = np.full_like(y_vals, np.nan, dtype=float)
            fit[valid] = np.polyval(coeffs, x_clean)

            ss_res = np.sum((y_clean - np.polyval(coeffs, x_clean))**2)
            ss_tot = np.sum((y_clean - np.mean(y_clean))**2)

            r2 = 1 - ss_res/ss_tot if ss_tot != 0 else np.nan
        else:
            fit = y_vals
            r2 = np.nan


        

        fig_auc = go.Figure()

        fig_auc.add_trace(go.Scatter(
            x=df_auc["time"],
            y=y_vals,
            name=f"AUC (R²={r2:.3f})",
            mode="lines+markers"
        ))

        fig_auc.add_trace(go.Scatter(
            x=df_auc["time"],
            y=fit,
            name="Fit",
            line=dict(dash="dash")
        ))

        st.plotly_chart(fig_auc, use_container_width=True)
