  # v 01/07/2026 FINAL STABLE (APIES + REGRESSION + CLEAN AUC)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import re

from io import BytesIO

def dataframe_to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False)
    return output.getvalue()


st.set_page_config(page_title='SpectraKinetics v17', layout='wide')

page = st.sidebar.radio("Navigation",
                        ["APIES Dashboard", "AUC Analysis", "Kinetics Mode"])


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
# ---------- CASE KINSPEC TIME SERIES ----------
    for i, line in enumerate(content):
        if "kinetic time" in line.lower():

            # extract time values
            time_parts = re.split(r"\s+|\t+", content[i].strip())

            times = []
            for val in time_parts[1:]:
                try:
                    times.append(float(val))
                except:
                    continue

            wavelengths = []
            matrix = []

            # read data rows
            for line in content[i+1:]:
                parts = re.split(r"\s+|\t+", line.strip())

                if len(parts) < len(times) + 1:
                    continue

                try:
                    # first column = wavelength (or index)
                    wavelengths.append(float(parts[1]))

                    row = []
                    for j in range(len(times)):
                        row.append(float(parts[j + 2]))

                    matrix.append(row)

                except:
                    continue

            wavelengths = np.array(wavelengths)
            matrix = np.array(matrix)

            # transpose → make time behave like excitation
            spectra = {}
            for j, t in enumerate(times):
                spectra[t] = matrix[:, j]

            return {
                "wavelengths": wavelengths,
                "spectra": spectra,
                "filename": filename
                "mode": "kinetic"
            }
        
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
                "mode": "spectral"

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
            "mode": "spectral"

        }


    # ---------- CASE 2: ROBUST IFEPEM FORMAT ----------
    header = re.split(r"\s+|\t+", content[0].strip())

    # ✅ extract ONLY numeric excitation values
    ex_vals = []
    for val in header:
        try:
            ex_vals.append(float(val))
        except:
            continue

    if len(ex_vals) > 0:

        wavelengths, matrix = [], []

    # ✅ find first numeric data row (skip "nm" or junk rows)
        start_idx = 1
        while start_idx < len(content):
            parts = re.split(r"\s+|\t+", content[start_idx].strip())
            try:
                float(parts[0])
                break
            except:
                start_idx += 1

        for line in content[start_idx:]:
            parts = re.split(r"\s+|\t+", line.strip())

            if len(parts) < len(ex_vals) + 1:
                continue

            try:
                wavelengths.append(float(parts[0]))

            # ✅ ensure correct column alignment
                row = []
                for i in range(len(ex_vals)):
                    row.append(float(parts[i+1]))

                matrix.append(row)

            except:
                continue

        wavelengths = np.array(wavelengths)
        matrix = np.array(matrix)

    # ✅ FIX: sort excitation values ascending
        ex_vals = np.array(ex_vals)
        sort_idx = np.argsort(ex_vals)

        ex_vals = ex_vals[sort_idx]
        matrix = matrix[:, sort_idx]

        spectra = {}
        for j, ex in enumerate(ex_vals):
            spectra[ex] = matrix[:, j]
    
        return {
            "wavelengths": wavelengths,
            "spectra": spectra,
            "filename": filename
            "mode": "spectral"

        }


    # ---------- FALLBACK ----------
    return {
        "wavelengths": np.array([]),
        "spectra": {},
        "filename": filename
        "mode": "spectral"

    }


def apply_ife_correction(pem, abs_data):

    wl_em = pem["wavelengths"]
    spectra = pem["spectra"]

    wl_abs = abs_data["wavelengths"]
    abs_vals = list(abs_data["spectra"].values())[0]

    # ✅ ensure increasing wavelength for interpolation
    if wl_abs[0] > wl_abs[-1]:
        wl_abs = wl_abs[::-1]
        abs_vals = abs_vals[::-1]

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
        "mode": pem.get("mode", "unknown")
    }

if not data:
    st.info("Upload data to begin.")
    st.stop()
  
# ✅ build excitation selector dynamically
all_ex = sorted({
    int(round(ex))
    for d in data.values()
    for ex in d["spectra"].keys()
})


ex_toggle = st.sidebar.selectbox(
    "Excitation Wavelength (nm)",
    all_ex
)



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
        wl = d["wavelengths"]
      
        # --- Get absorbance data for Aggregation Index ---
        abs_pair = data_raw.get(name, {})
        
        if "abs" in abs_pair:
            wl_abs = abs_pair["abs"]["wavelengths"]
            abs_vals = list(abs_pair["abs"]["spectra"].values())[0]
        
            if wl_abs[0] > wl_abs[-1]:
                wl_abs = wl_abs[::-1]
                abs_vals = abs_vals[::-1]
        
            A280 = np.interp(280, wl_abs, abs_vals)
            A350 = np.interp(350, wl_abs, abs_vals)
        
            denom = A280 - A350
            agg_index = (A350 / denom) * 100 if abs(denom) > 1e-9 else np.nan
        else:
            agg_index = np.nan

        
# match nearest excitation (handles float issues)
        ex_actual = min(d["spectra"].keys(), key=lambda k: abs(k - ex_toggle))
        y = d["spectra"][ex_actual]



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
            "Aggregation Index": agg_index,
            "AUC IR": auc_ir,
            "AUC IF": auc_if
        })

    
    
    df = pd.DataFrame(rows)

    if not df.empty and "Time" in df.columns:
        df = df.sort_values(by="Time", na_position="last").reset_index(drop=True)
    
    if df.empty:
        st.warning("No valid datasets for selected excitation wavelength.")
        st.stop()


    st.dataframe(df, use_container_width=True)

    excel_data = dataframe_to_excel(df)
    
    st.download_button(
        label="Download APIES Metrics (Excel)",
        data=excel_data,
        file_name="APIES_metrics.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    # ✅ Spectra overlay
    st.subheader("Spectra Overlay")
    fig_spec = go.Figure()
  
    
    for name, d in data.items():
        ex_actual = min(d["spectra"].keys(), key=lambda k: abs(k - ex_toggle))
    
        fig_spec.add_trace(go.Scatter(
            x=d["wavelengths"],
            y=d["spectra"][ex_actual],
            name=name
        ))

    fig_spec.update_layout(
        title=f"Fluorescence Emission Spectra (Ex = {ex_toggle} nm)",
        xaxis_title="Emission Wavelength (nm)",
        yaxis_title="Fluorescence Intensity (Counts/µA)",
        template="plotly_white"
    )
    
    st.plotly_chart(fig_spec, use_container_width=True)
 # aBSORBANCE
    st.subheader("Absorbance Overlay")
    
    fig_abs = go.Figure()
    
    for name, pair in data_raw.items():
        if "abs" not in pair:
            continue
    
        abs_data = pair["abs"]
    
        wl_abs = abs_data["wavelengths"]
        abs_vals = list(abs_data["spectra"].values())[0]
    
        # ensure sorted
        if wl_abs[0] > wl_abs[-1]:
            wl_abs = wl_abs[::-1]
            abs_vals = abs_vals[::-1]
    
        fig_abs.add_trace(go.Scatter(
            x=wl_abs,
            y=abs_vals,
            name=name
        ))
    
    fig_abs.update_layout(
        title="Absorbance Spectra",
        xaxis_title="Wavelength (nm)",
        yaxis_title="Absorbance",
        template="plotly_white"
    )
    
    st.plotly_chart(fig_abs, use_container_width=True)

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
    
    df_reg = pd.DataFrame({
        "Time / Index": x_vals,
        "IR/IF (AUC)": df["IR/IF (AUC)"],
        "Fit": fit,
        "I350/I330": df["I350/I330"]
    })
    
    excel_reg = dataframe_to_excel(df_reg)
    
    st.download_button(
        label="Download Regression Data (Excel)",
        data=excel_reg,
        file_name="APIES_regression.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    
    # =====================
    # ✅ PAPER-STYLE APIES PLOTS (IMPROVED)
    # =====================
    from plotly.subplots import make_subplots
    
    st.subheader("APIES Metrics vs Time")
    
    fig_stack = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.15,   # ✅ increase spacing
        subplot_titles=(
            "IR/IF Ratio",
            "I350/I330 Ratio",
            "Aggregation Index"
        )
    )
    
    # --- IR/IF ---
    fig_stack.add_trace(
        go.Scatter(
            x=x_vals,
            y=df["IR/IF (AUC)"],
            mode="lines+markers",
            name="IR/IF"
        ),
        row=1, col=1
    )
    
    # --- I350/I330 ---
    fig_stack.add_trace(
        go.Scatter(
            x=x_vals,
            y=df["I350/I330"],
            mode="lines+markers",
            name="I350/I330"
        ),
        row=2, col=1
    )
    
    # --- Aggregation Index ---
    fig_stack.add_trace(
        go.Scatter(
            x=x_vals,
            y=df["Aggregation Index"],
            mode="lines+markers",
            name="Agg Index"
        ),
        row=3, col=1
    )
    
    # ✅ Axis labels + units
    fig_stack.update_yaxes(title_text="IR/IF ", row=1, col=1)
    fig_stack.update_yaxes(title_text="I350/I330 ", row=2, col=1)
    fig_stack.update_yaxes(title_text="Agg Index (%)", row=3, col=1)
    
    # ✅ Only bottom x-axis gets label
    fig_stack.update_xaxes(title_text="Time / Sample Index", row=3, col=1)
    
    fig_stack.update_layout(
        height=900,              # ✅ taller to avoid crowding
        template="plotly_white",
        showlegend=False
    )
    
    st.plotly_chart(fig_stack, use_container_width=True)

    # Prepare export data
    df_export = pd.DataFrame({
        "Time / Index": x_vals,
        "IR/IF (AUC)": df["IR/IF (AUC)"],
        "I350/I330": df["I350/I330"],
        "Aggregation Index (%)": df["Aggregation Index"]
    })
    
    excel_stack = dataframe_to_excel(df_export)
    
    st.download_button(
        label="Download APIES Plot Data (Excel)",
        data=excel_stack,
        file_name="APIES_plot_data.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# =====================
# ✅ AUC ANALYSIS (FIXED)
# =====================
if page == "AUC Analysis":

    st.title("AUC Analysis")

    selected = st.selectbox("Dataset", list(data.keys()))
    d = data[selected]

    ex_actual = min(d["spectra"].keys(), key=lambda k: abs(k - ex_toggle))

    wl = d["wavelengths"]
    y = d["spectra"][ex_actual]

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

            ex_actual = min(dataset["spectra"].keys(), key=lambda k: abs(k - ex_toggle))

            wl_f = dataset["wavelengths"]
            y_f = dataset["spectra"][ex_actual]

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

        excel_auc_analysis = dataframe_to_excel(df_auc)
        
        st.download_button(
            label="Download AUC Analysis (Excel)",
            data=excel_auc_analysis,
            file_name="AUC_analysis.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

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

        df_auc_export = df_auc.copy() 
        excel_auc = dataframe_to_excel(df_auc_export)
        
        st.download_button(
            label="Download AUC Results (Excel)",
            data=excel_auc,
            file_name="AUC_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

# =====================
# ✅ TRUE KINETICS MODE
# =====================
if page == "Kinetics Mode":

    st.title("Kinetics Analysis (Intensity vs Time)")

    # ✅ filter only kinetic datasets
    kinetic_data = {
        name: d for name, d in data.items()
        if d.get("mode") == "kinetic"
    }

    if not kinetic_data:
        st.warning("No kinetic (KinSpec) datasets loaded.")
        st.stop()

    selected = st.selectbox("Dataset", list(kinetic_data.keys()))
    d = kinetic_data[selected]

    wl = d["wavelengths"]
    spectra = d["spectra"]   # keys = time, values = spectra

    times = np.array(sorted(spectra.keys()))

    # ✅ wavelength selection
    selected_wl = st.slider(
        "Select Emission Wavelength (nm)",
        float(min(wl)),
        float(max(wl)),
        float(np.median(wl))
    )

    # ✅ extract intensity vs time
    intensities = []

    idx = np.argmin(np.abs(wl - selected_wl))

    for t in times:
        intensities.append(spectra[t][idx])

    intensities = np.array(intensities)

    # ✅ main plot
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=times,
        y=intensities,
        mode="lines+markers",
        name=f"{selected_wl:.1f} nm"
    ))

    fig.update_layout(
        title=f"Kinetics at {selected_wl:.1f} nm",
        xaxis_title="Time (s)",
        yaxis_title="Fluorescence Intensity",
        template="plotly_white"
    )

    st.plotly_chart(fig, use_container_width=True)

    # =====================
    # ✅ MULTI-WAVELENGTH TRACKING
    # =====================
    st.subheader("Multi-Wavelength Tracking")

    wl_points = st.text_input(
        "Enter wavelengths (comma-separated)",
        "330,350,370"
    )

    try:
        wl_list = [float(x.strip()) for x in wl_points.split(",")]

        fig_multi = go.Figure()

        for w in wl_list:

            idx = np.argmin(np.abs(wl - w))
            series = [spectra[t][idx] for t in times]

            fig_multi.add_trace(go.Scatter(
                x=times,
                y=series,
                mode="lines",
                name=f"{w} nm"
            ))

        fig_multi.update_layout(
            title="Kinetics at Multiple Wavelengths",
            xaxis_title="Time (s)",
            yaxis_title="Intensity",
            template="plotly_white"
        )

        st.plotly_chart(fig_multi, use_container_width=True)

    except:
        st.info("Enter valid numeric wavelengths")

    # =====================
    # ✅ EXPORT DATA
    # =====================
    df_out = pd.DataFrame({"Time (s)": times})

    # add selected wavelength
    df_out[f"{selected_wl:.1f} nm"] = intensities

    excel_bytes = dataframe_to_excel(df_out)

    st.download_button(
        label="Download Kinetics Data",
        data=excel_bytes,
        file_name="kinetics_data.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

