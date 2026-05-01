import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import os
import io

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="APIES Dashboard",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@400;600&display=swap');

:root {
    --bg: #f4f6f9;
    --surface: #ffffff;
    --surface2: #eef1f5;
    --border: #d0d7e2;
    --accent: #0e6b8c;
    --accent-light: #e0f3f9;
    --accent2: #0a9e7a;
    --text: #1a2332;
    --muted: #5a6a7e;
    --danger: #c0392b;
    --warn: #e67e22;
    --shadow: 0 1px 4px rgba(14,107,140,0.08), 0 2px 12px rgba(14,107,140,0.06);
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: var(--bg);
    color: var(--text);
}

.stApp { background-color: var(--bg); }

h1, h2, h3 { font-family: 'Inter', sans-serif; font-weight: 600; color: var(--text); }

.hero {
    background: linear-gradient(120deg, #0e6b8c 0%, #0a5a78 60%, #0a9e7a 100%);
    border-radius: 14px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 4px 24px rgba(14,107,140,0.18);
    position: relative;
}
.hero h1 {
    font-size: 1.75rem;
    margin: 0 0 0.3rem 0;
    color: #ffffff;
    font-weight: 700;
    letter-spacing: -0.02em;
}
.hero p { color: rgba(255,255,255,0.8); margin: 0; font-size: 0.95rem; }
.nbl-badge {
    position: absolute;
    top: 1.25rem;
    right: 1.75rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.4rem;
    font-weight: 600;
    color: #0a2a4a;
    letter-spacing: 0.18em;
    opacity: 0.85;
}

.card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1rem;
    box-shadow: var(--shadow);
}
.card-title {
    font-family: 'Inter', sans-serif;
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--accent);
    margin-bottom: 0.75rem;
}

.stat-row { display: flex; gap: 0.75rem; margin-bottom: 1rem; flex-wrap: wrap; }
.stat {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 0.75rem 1rem;
    flex: 1;
    min-width: 120px;
    box-shadow: var(--shadow);
}
.stat-label { font-size: 0.68rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.08em; font-weight: 500; margin-bottom: 0.2rem; }
.stat-value { font-family: 'IBM Plex Mono', monospace; font-size: 1rem; color: var(--accent); font-weight: 600; }

.file-tag {
    display: inline-block;
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 0.2rem 0.6rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    color: var(--muted);
    margin: 0.15rem;
}
.file-tag.active {
    background: var(--accent-light);
    border-color: var(--accent);
    color: var(--accent);
    font-weight: 600;
}

.stButton > button {
    background: var(--accent) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.85rem !important;
    font-weight: 600 !important;
    padding: 0.5rem 1.2rem !important;
    transition: all 0.15s !important;
    box-shadow: 0 2px 8px rgba(14,107,140,0.2) !important;
}
.stButton > button:hover {
    background: #0a5a78 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 16px rgba(14,107,140,0.3) !important;
}

[data-testid="stSidebar"] {
    background-color: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}

div[data-testid="stExpander"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
}

.stTabs [data-baseweb="tab-list"] {
    background: var(--surface2) !important;
    border-radius: 8px !important;
    padding: 3px !important;
    gap: 2px !important;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 6px !important;
    color: var(--muted) !important;
    font-weight: 500 !important;
}
.stTabs [aria-selected="true"] {
    background: var(--surface) !important;
    color: var(--accent) !important;
    font-weight: 600 !important;
    box-shadow: var(--shadow) !important;
}
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
if "datasets" not in st.session_state:
    st.session_state.datasets = {}   # filename → {times, wavelengths, intensity_df}
if "active_file" not in st.session_state:
    st.session_state.active_file = None

# ── Parsing helper ─────────────────────────────────────────────────────────────
def parse_spectro_file(file_bytes, filename):
    """
    Format:
      Rows 1-11  : metadata
      Row 5      : labels — scan for 'Absorbance [a.u.]' to find absorbance column index
      Row 12     : col A (ignored), col B (ignored), col C onward = time values
                   (stops at first empty/non-numeric cell)
      Rows 13+   : col B = wavelength, cols C..C+n = intensity,
                   absorbance_col_idx = absorbance values (if present)
    """
    try:
        content = file_bytes.decode("utf-8", errors="replace")
    except Exception:
        content = file_bytes.decode("latin-1", errors="replace")

    # Strip BOM if present
    if content.startswith("\ufeff"):
        content = content[1:]

    lines = content.splitlines()

    if len(lines) < 13:
        return None, f"File too short ({len(lines)} lines)"

    # ── Detect absorbance column from Row 5 (index 4) ────────────────────────
    label_row = lines[4].split("\t")
    abs_col_idx = None
    for j, cell in enumerate(label_row):
        if "absorbance" in cell.strip().lower() and "%" not in cell.lower():
            abs_col_idx = j
            break

    # ── Row 12 (index 11) → time values starting at col C (index 2) ─────────
    time_row = lines[11].split("\t")
    times = []
    for val in time_row[2:]:
        val = val.strip()
        if val == "":
            break
        try:
            times.append(float(val))
        except ValueError:
            break

    if not times:
        return None, "Could not parse time values from row 12"

    # ── Rows 13+ → wavelength, intensities, and absorbance ───────────────────
    wavelengths = []
    intensity_matrix = []
    absorbance_values = []

    for line in lines[12:]:
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        wl_str = parts[1].strip()
        if wl_str == "":
            continue
        try:
            wl = float(wl_str)
        except ValueError:
            continue

        row_intensities = []
        for val in parts[2: 2 + len(times)]:
            try:
                row_intensities.append(float(val.strip()))
            except ValueError:
                row_intensities.append(np.nan)

        # Pad if shorter
        while len(row_intensities) < len(times):
            row_intensities.append(np.nan)

        wavelengths.append(wl)
        intensity_matrix.append(row_intensities)

        # Absorbance value for this wavelength row
        if abs_col_idx is not None and abs_col_idx < len(parts):
            try:
                absorbance_values.append(float(parts[abs_col_idx].strip()))
            except ValueError:
                absorbance_values.append(np.nan)
        else:
            absorbance_values.append(np.nan)

    if not wavelengths:
        return None, "No wavelength data found"

    intensity_df = pd.DataFrame(
        intensity_matrix,
        index=wavelengths,
        columns=times
    )
    intensity_df.index.name = "wavelength_nm"
    intensity_df.columns.name = "time_s"

    # Build absorbance Series indexed by wavelength
    absorbance_series = pd.Series(absorbance_values, index=wavelengths, name="absorbance")
    has_absorbance = abs_col_idx is not None and not absorbance_series.isna().all()

    return {
        "times": np.array(times),
        "wavelengths": np.array(wavelengths),
        "intensity": intensity_df,
        "absorbance": absorbance_series,
        "has_absorbance": has_absorbance,
        "filename": filename,
    }, None


def nearest(arr, val):
    arr = np.array(arr)
    idx = np.argmin(np.abs(arr - val))
    return arr[idx], idx


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="card-title">📁 Import Files</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Upload .txt / .tsv files",
        type=["txt", "tsv"],
        accept_multiple_files=True,
        help="Select all files from your experiment folder",
    )

    if uploaded:
        new_count = 0
        err_count = 0
        for f in uploaded:
            if f.name not in st.session_state.datasets:
                data, err = parse_spectro_file(f.read(), f.name)
                if data:
                    st.session_state.datasets[f.name] = data
                    new_count += 1
                else:
                    st.error(f"❌ {f.name}: {err}")
                    err_count += 1

        if new_count:
            st.success(f"✅ Loaded {new_count} file(s)")
            if not st.session_state.active_file:
                st.session_state.active_file = list(st.session_state.datasets.keys())[0]

    st.divider()

    if st.session_state.datasets:
        st.markdown('<div class="card-title">🗂 Active File</div>', unsafe_allow_html=True)
        st.session_state.active_file = st.selectbox(
            "Select file",
            options=list(st.session_state.datasets.keys()),
            index=list(st.session_state.datasets.keys()).index(st.session_state.active_file)
            if st.session_state.active_file in st.session_state.datasets else 0,
            label_visibility="collapsed",
        )

        ds = st.session_state.datasets[st.session_state.active_file]

        st.divider()
        st.markdown('<div class="card-title">⚙️ Analysis Parameters</div>', unsafe_allow_html=True)

        wl_min = float(ds["wavelengths"].min())
        wl_max = float(ds["wavelengths"].max())
        wl_interest = st.number_input(
            "Wavelength of interest (nm)",
            min_value=wl_min,
            max_value=wl_max,
            value=min(350.0, wl_max),
            step=0.5,
            format="%.1f",
        )

        t_min = float(ds["times"].min())
        t_max = float(ds["times"].max())
        st.markdown("**Time range of interest (s)**")
        t_start = st.number_input("Start (s)", min_value=t_min, max_value=t_max, value=t_min, step=1.0, format="%.2f")
        t_end   = st.number_input("End (s)",   min_value=t_min, max_value=t_max, value=t_max, step=1.0, format="%.2f")

        st.divider()
        st.markdown('<div class="card-title">📊 Spectra Parameters</div>', unsafe_allow_html=True)
        t_spectra = st.number_input(
            "Time point for spectra (s)",
            min_value=t_min,
            max_value=t_max,
            value=t_min,
            step=1.0,
            format="%.2f",
        )

        if st.button("🗑 Clear All Files"):
            st.session_state.datasets = {}
            st.session_state.active_file = None
            st.rerun()

    else:
        wl_interest = 350.0
        t_start, t_end = 0.0, 100.0
        t_spectra = 0.0


# ── Main ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="nbl-badge">NBL</div>
  <h1>APIES Dashboard</h1>
  <p>Absorbance · Polarized Intrinsic Emission · Scattering</p>
</div>
""", unsafe_allow_html=True)

if not st.session_state.datasets:
    st.markdown("""
    <div class="card" style="text-align:center; padding: 3rem;">
        <div style="font-size:3rem; margin-bottom:1rem;">📂</div>
        <div style="font-family:'Space Mono',monospace; color:var(--accent); font-size:1rem; margin-bottom:0.5rem;">No files loaded</div>
        <div style="color:var(--muted); font-size:0.9rem;">Upload your .txt experiment files using the sidebar to get started.</div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

ds = st.session_state.datasets[st.session_state.active_file]
times = ds["times"]
wavelengths = ds["wavelengths"]
intensity = ds["intensity"]

# Stats row
wl_actual, _ = nearest(wavelengths, wl_interest)
t_actual_spec, _ = nearest(times, t_spectra)

st.markdown(f"""
<div class="stat-row">
  <div class="stat">
    <div class="stat-label">Files Loaded</div>
    <div class="stat-value">{len(st.session_state.datasets)}</div>
  </div>
  <div class="stat">
    <div class="stat-label">Wavelength Range</div>
    <div class="stat-value">{wavelengths.min():.0f}–{wavelengths.max():.0f} nm</div>
  </div>
  <div class="stat">
    <div class="stat-label">Time Range</div>
    <div class="stat-value">{times.min():.0f}–{times.max():.0f} s</div>
  </div>
  <div class="stat">
    <div class="stat-label">Time Points</div>
    <div class="stat-value">{len(times)}</div>
  </div>
  <div class="stat">
    <div class="stat-label">Nearest λ</div>
    <div class="stat-value">{wl_actual:.2f} nm</div>
  </div>
</div>
""", unsafe_allow_html=True)

# File tags
tags_html = "".join(
    f'<span class="file-tag {"active" if k == st.session_state.active_file else ""}">{k}</span>'
    for k in st.session_state.datasets
)
st.markdown(f'<div style="margin-bottom:1.5rem;">{tags_html}</div>', unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_kinetics, tab_spectra, tab_analysis, tab_data = st.tabs(["📈 Kinetics", "🌈 Spectra", "🧪 Spectral Analysis", "🗃 Raw Data"])

# ── KINETICS TAB ──────────────────────────────────────────────────────────────
with tab_kinetics:
    col1, col2 = st.columns([3, 1])
    with col2:
        st.markdown('<div class="card-title">Overlay Options</div>', unsafe_allow_html=True)
        overlay_all = st.checkbox("Overlay all loaded files", value=False)
        show_markers = st.checkbox("Show data markers", value=False)
        smooth = st.checkbox("Smooth (rolling avg)", value=False)
        if smooth:
            window = st.slider("Window size", 3, 51, 5, step=2)

    with col1:
        fig = go.Figure()

        files_to_plot = list(st.session_state.datasets.keys()) if overlay_all else [st.session_state.active_file]
        colors = px.colors.qualitative.Vivid

        for i, fname in enumerate(files_to_plot):
            d = st.session_state.datasets[fname]
            wl_act, _ = nearest(d["wavelengths"], wl_interest)

            # Slice time range
            mask = (d["times"] >= t_start) & (d["times"] <= t_end)
            t_plot = d["times"][mask]
            y_plot = d["intensity"].loc[wl_act].values[mask]

            if smooth:
                y_plot = pd.Series(y_plot).rolling(window, center=True, min_periods=1).mean().values

            fig.add_trace(go.Scatter(
                x=t_plot,
                y=y_plot,
                mode="lines+markers" if show_markers else "lines",
                name=fname,
                line=dict(color=colors[i % len(colors)], width=2),
                marker=dict(size=4),
            ))

        # Highlight vertical line at t_spectra
        fig.add_vline(
            x=t_actual_spec,
            line_dash="dash",
            line_color="#0a9e7a",
            annotation_text=f"t={t_actual_spec:.1f}s",
            annotation_font_color="#0a9e7a",
        )

        fig.update_layout(
            template="plotly_white",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#ffffff",
            title=dict(
                text=f"Kinetics at λ = {wl_actual:.2f} nm (Ex 280 nm)",
                font=dict(family="Inter", size=14, color="#0e6b8c"),
            ),
            xaxis=dict(title="Time (s)", gridcolor="#eef1f5", linecolor="#d0d7e2", tickfont=dict(color="#5a6a7e")),
            yaxis=dict(title="Intensity (a.u.)", gridcolor="#eef1f5", linecolor="#d0d7e2", tickfont=dict(color="#5a6a7e")),
            legend=dict(bgcolor="rgba(255,255,255,0.9)", bordercolor="#d0d7e2", borderwidth=1),
            margin=dict(t=50, b=40, l=50, r=20),
            hovermode="x unified",
        )

        st.plotly_chart(fig, use_container_width=True)

    # Highlighted data snippet
    with st.expander("🔍 Highlighted data at selected wavelength"):
        mask = (times >= t_start) & (times <= t_end)
        slice_df = pd.DataFrame({
            "Time (s)": times[mask],
            f"Intensity @ {wl_actual:.2f} nm": intensity.loc[wl_actual].values[mask],
        })
        st.dataframe(
            slice_df.style.highlight_max(subset=[f"Intensity @ {wl_actual:.2f} nm"], color="#003d33")
                         .highlight_min(subset=[f"Intensity @ {wl_actual:.2f} nm"], color="#3d0000"),
            use_container_width=True,
            height=250,
        )

# ── SPECTRA TAB ───────────────────────────────────────────────────────────────
with tab_spectra:
    col1, col2 = st.columns([3, 1])
    with col2:
        st.markdown('<div class="card-title">Spectra Options</div>', unsafe_allow_html=True)
        multi_t = st.checkbox("Plot multiple time points", value=False)
        if multi_t:
            t_list_str = st.text_input("Time points (comma-separated, s)", value="0, 500, 1000, 2000")
            try:
                t_list = [float(x.strip()) for x in t_list_str.split(",")]
            except Exception:
                t_list = [float(times[0])]
        else:
            t_list = [t_spectra]

        wl_range = st.slider(
            "Wavelength range (nm)",
            float(wavelengths.min()), float(wavelengths.max()),
            (float(wavelengths.min()), float(wavelengths.max())),
        )

    with col1:
        fig2 = go.Figure()
        colors = px.colors.sequential.Viridis

        for i, t_pt in enumerate(t_list):
            t_act, _ = nearest(times, t_pt)
            wl_mask = (wavelengths >= wl_range[0]) & (wavelengths <= wl_range[1])
            wl_plot = wavelengths[wl_mask]
            y_plot = intensity.loc[wl_plot, t_act].values

            color = colors[int(i / max(len(t_list) - 1, 1) * (len(colors) - 1))]
            fig2.add_trace(go.Scatter(
                x=wl_plot,
                y=y_plot,
                mode="lines",
                name=f"t = {t_act:.1f} s",
                line=dict(color=color, width=2),
                fill="tozeroy" if not multi_t else None,
                fillcolor="rgba(0,212,170,0.07)" if not multi_t else None,
            ))

        # Mark wavelength of interest
        fig2.add_vline(
            x=wl_actual,
            line_dash="dot",
            line_color="#0e6b8c",
            annotation_text=f"λ={wl_actual:.1f}nm",
            annotation_font_color="#0e6b8c",
        )

        fig2.update_layout(
            template="plotly_white",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#ffffff",
            title=dict(
                text="Emission Spectra (Ex 280 nm)",
                font=dict(family="Inter", size=14, color="#0e6b8c"),
            ),
            xaxis=dict(title="Emission Wavelength (nm)", gridcolor="#eef1f5", linecolor="#d0d7e2", tickfont=dict(color="#5a6a7e")),
            yaxis=dict(title="Intensity (a.u.)", gridcolor="#eef1f5", linecolor="#d0d7e2", tickfont=dict(color="#5a6a7e")),
            legend=dict(bgcolor="rgba(255,255,255,0.9)", bordercolor="#d0d7e2", borderwidth=1),
            margin=dict(t=50, b=40, l=50, r=20),
        )

        st.plotly_chart(fig2, use_container_width=True)

    with st.expander("🔍 Spectra data table"):
        t_act_spec, _ = nearest(times, t_spectra)
        spec_df = pd.DataFrame({
            "Wavelength (nm)": wavelengths,
            f"Intensity @ t={t_act_spec:.1f}s": intensity[t_act_spec].values,
        })
        st.dataframe(spec_df, use_container_width=True, height=250)

# ── SPECTRAL ANALYSIS TAB ─────────────────────────────────────────────────────
with tab_analysis:

    st.markdown('<div class="card-title">Spectral Analysis Tools</div>', unsafe_allow_html=True)

    # ── Helper: get intensity series for a wavelength over time ──
    def get_intensity_series(ds, wl_target, t_mask=None):
        wl_act, _ = nearest(ds["wavelengths"], wl_target)
        vals = ds["intensity"].loc[wl_act].values
        t = ds["times"]
        if t_mask is not None:
            vals = vals[t_mask]
            t = t[t_mask]
        return t, vals, wl_act

    analysis_mask = (times >= t_start) & (times <= t_end)

    # ── TOOL 1: IR/IF Ratio (Rayleigh-Mie scattering) ────────────────────────
    with st.expander("📊 Tool 1 — IR/IF Ratio (Rayleigh-Mie Scattering)", expanded=True):
        st.markdown("""
        <div style="color:var(--muted); font-size:0.85rem; margin-bottom:1rem;">
        Ratio of scattered light (IR, Rayleigh-Mie band) to intrinsic fluorescence (IF).
        Rising IR/IF over time indicates increasing aggregation/scattering.
        </div>
        """, unsafe_allow_html=True)

        col_ctrl, col_vals = st.columns([2, 1])
        with col_ctrl:
            ir_wl = st.number_input("IR wavelength (nm)", value=280.0, step=0.5, format="%.1f", key="ir_wl")
            if_wl = st.number_input("IF wavelength (nm)", value=340.0, step=0.5, format="%.1f", key="if_wl")

        t_ir, ir_vals, ir_actual = get_intensity_series(ds, ir_wl, analysis_mask)
        _,    if_vals, if_actual = get_intensity_series(ds, if_wl, analysis_mask)

        # Avoid divide by zero
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio_irif = np.where(if_vals != 0, ir_vals / if_vals, np.nan)

        current_irif = ratio_irif[-1] if not np.all(np.isnan(ratio_irif)) else np.nan
        mean_irif    = np.nanmean(ratio_irif)

        with col_vals:
            st.markdown(f"""
            <div class="card" style="margin-top:0.5rem;">
                <div class="stat-label">Current IR/IF</div>
                <div class="stat-value" style="font-size:1.4rem;">{current_irif:.4f}</div>
                <div class="stat-label" style="margin-top:0.5rem;">Mean IR/IF</div>
                <div class="stat-value">{mean_irif:.4f}</div>
                <div class="stat-label" style="margin-top:0.5rem;">IR λ used</div>
                <div class="stat-value">{ir_actual:.2f} nm</div>
                <div class="stat-label" style="margin-top:0.5rem;">IF λ used</div>
                <div class="stat-value">{if_actual:.2f} nm</div>
            </div>
            """, unsafe_allow_html=True)

        fig_irif = go.Figure()
        fig_irif.add_trace(go.Scatter(
            x=t_ir, y=ratio_irif,
            mode="lines",
            name="IR/IF ratio",
            line=dict(color="#0e6b8c", width=2),
            fill="tozeroy",
            fillcolor="rgba(14,107,140,0.07)",
        ))
        fig_irif.update_layout(
            template="plotly_white",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#ffffff",
            title=dict(text=f"IR/IF Ratio over Time  (IR={ir_actual:.1f} nm / IF={if_actual:.1f} nm)",
                       font=dict(family="Inter", size=13, color="#0e6b8c")),
            xaxis=dict(title="Time (s)", gridcolor="#eef1f5", linecolor="#d0d7e2"),
            yaxis=dict(title="IR/IF Ratio", gridcolor="#eef1f5", linecolor="#d0d7e2"),
            margin=dict(t=45, b=40, l=50, r=20),
            hovermode="x unified",
        )
        st.plotly_chart(fig_irif, use_container_width=True)

        with st.expander("🔍 IR/IF data table"):
            irif_df = pd.DataFrame({
                "Time (s)": t_ir,
                f"IR @ {ir_actual:.2f} nm": ir_vals,
                f"IF @ {if_actual:.2f} nm": if_vals,
                "IR/IF Ratio": ratio_irif,
            })
            st.dataframe(irif_df, use_container_width=True, height=220)
            csv_irif = irif_df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇ Download IR/IF data", csv_irif, "irif_ratio.csv", "text/csv", key="dl_irif")

    # ── TOOL 2: I350/I330 Ratio (Polarized Intrinsic Emission) ───────────────
    with st.expander("📊 Tool 2 — I350/I330 Ratio (Polarized Intrinsic Emission)", expanded=True):
        st.markdown("""
        <div style="color:var(--muted); font-size:0.85rem; margin-bottom:1rem;">
        Ratio of emission intensity at 350 nm to 330 nm. Reflects changes in the
        local environment of tryptophan residues — a red shift (rising ratio) indicates
        increased solvent exposure, unfolding, or conformational change.
        </div>
        """, unsafe_allow_html=True)

        col_ctrl2, col_vals2 = st.columns([2, 1])
        with col_ctrl2:
            wl_350_input = st.number_input("Numerator wavelength (nm)", value=350.0, step=0.5, format="%.1f", key="wl350")
            wl_330_input = st.number_input("Denominator wavelength (nm)", value=330.0, step=0.5, format="%.1f", key="wl330")

        t_350, i350_vals, wl_350_actual = get_intensity_series(ds, wl_350_input, analysis_mask)
        _,     i330_vals, wl_330_actual = get_intensity_series(ds, wl_330_input, analysis_mask)

        with np.errstate(divide="ignore", invalid="ignore"):
            ratio_pie = np.where(i330_vals != 0, i350_vals / i330_vals, np.nan)

        current_pie = ratio_pie[-1] if not np.all(np.isnan(ratio_pie)) else np.nan
        mean_pie    = np.nanmean(ratio_pie)

        with col_vals2:
            st.markdown(f"""
            <div class="card" style="margin-top:0.5rem;">
                <div class="stat-label">Current I350/I330</div>
                <div class="stat-value" style="font-size:1.4rem;">{current_pie:.4f}</div>
                <div class="stat-label" style="margin-top:0.5rem;">Mean I350/I330</div>
                <div class="stat-value">{mean_pie:.4f}</div>
                <div class="stat-label" style="margin-top:0.5rem;">λ₁ used</div>
                <div class="stat-value">{wl_350_actual:.2f} nm</div>
                <div class="stat-label" style="margin-top:0.5rem;">λ₂ used</div>
                <div class="stat-value">{wl_330_actual:.2f} nm</div>
            </div>
            """, unsafe_allow_html=True)

        fig_pie = go.Figure()
        fig_pie.add_trace(go.Scatter(
            x=t_350, y=ratio_pie,
            mode="lines",
            name="I350/I330",
            line=dict(color="#0a9e7a", width=2),
            fill="tozeroy",
            fillcolor="rgba(10,158,122,0.07)",
        ))
        fig_pie.update_layout(
            template="plotly_white",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#ffffff",
            title=dict(text=f"I{wl_350_actual:.0f}/I{wl_330_actual:.0f} Ratio over Time",
                       font=dict(family="Inter", size=13, color="#0a9e7a")),
            xaxis=dict(title="Time (s)", gridcolor="#eef1f5", linecolor="#d0d7e2"),
            yaxis=dict(title=f"I{wl_350_actual:.0f}/I{wl_330_actual:.0f} Ratio", gridcolor="#eef1f5", linecolor="#d0d7e2"),
            margin=dict(t=45, b=40, l=50, r=20),
            hovermode="x unified",
        )
        st.plotly_chart(fig_pie, use_container_width=True)

        with st.expander("🔍 I350/I330 data table"):
            pie_df = pd.DataFrame({
                "Time (s)": t_350,
                f"I @ {wl_350_actual:.2f} nm": i350_vals,
                f"I @ {wl_330_actual:.2f} nm": i330_vals,
                f"I{wl_350_actual:.0f}/I{wl_330_actual:.0f} Ratio": ratio_pie,
            })
            st.dataframe(pie_df, use_container_width=True, height=220)
            csv_pie = pie_df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇ Download I350/I330 data", csv_pie, "i350_i330_ratio.csv", "text/csv", key="dl_pie")

    # ── TOOL 3: Absorbance Analysis ───────────────────────────────────────────
    with st.expander("📊 Tool 3 — Absorbance Analysis (Concentration & Aggregation Index)", expanded=True):

        has_abs = ds.get("has_absorbance", False)
        abs_series = ds.get("absorbance", pd.Series(dtype=float))

        if has_abs:
            st.markdown("""
            <div style="color:#0a9e7a; font-size:0.85rem; margin-bottom:1rem;">
            ✅ Absorbance data detected in this file and loaded automatically.
            </div>
            """, unsafe_allow_html=True)

            # ── Absorbance spectrum plot ──────────────────────────────────
            fig_abs = go.Figure()
            fig_abs.add_trace(go.Scatter(
                x=abs_series.index,
                y=abs_series.values,
                mode="lines",
                name="Absorbance",
                line=dict(color="#0e6b8c", width=2),
                fill="tozeroy",
                fillcolor="rgba(14,107,140,0.07)",
            ))
            # Mark 280 and 350 nm
            for wl_mark, col_mark in [(280, "#0e6b8c"), (350, "#e67e22")]:
                wl_m, _ = nearest(abs_series.index.values, wl_mark)
                fig_abs.add_vline(x=wl_m, line_dash="dot", line_color=col_mark,
                                  annotation_text=f"{wl_m:.0f} nm",
                                  annotation_font_color=col_mark)
            fig_abs.update_layout(
                template="plotly_white",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="#ffffff",
                title=dict(text="Absorbance Spectrum", font=dict(family="Inter", size=13, color="#0e6b8c")),
                xaxis=dict(title="Wavelength (nm)", gridcolor="#eef1f5", linecolor="#d0d7e2"),
                yaxis=dict(title="Absorbance (a.u.)", gridcolor="#eef1f5", linecolor="#d0d7e2"),
                margin=dict(t=45, b=40, l=50, r=20),
            )
            st.plotly_chart(fig_abs, use_container_width=True)

        else:
            st.markdown("""
            <div style="color:var(--warn); font-size:0.85rem; margin-bottom:1rem;">
            ⚠️ No absorbance column detected in the current file.
            You can still enter values manually below.
            </div>
            """, unsafe_allow_html=True)

        st.divider()
        st.markdown("#### Protein Concentration (Beer-Lambert)")
        st.markdown("""
        <div style="color:var(--muted); font-size:0.82rem; margin-bottom:0.75rem;">
        c = A / (ε × l) — absorbance at your chosen wavelength is read from the file if available,
        or enter manually.
        </div>
        """, unsafe_allow_html=True)

        col_a, col_b, col_c, col_d = st.columns(4)
        with col_d:
            abs_wl = st.number_input("Measurement wavelength (nm)", value=280.0, step=0.5, format="%.1f", key="abs_wl")

        # Auto-fill absorbance value from parsed data if available
        if has_abs:
            abs_wl_nearest, _ = nearest(abs_series.index.values, abs_wl)
            auto_abs_val = float(abs_series.loc[abs_wl_nearest]) if not np.isnan(abs_series.loc[abs_wl_nearest]) else 0.0
        else:
            auto_abs_val = 0.500

        with col_a:
            abs_value = st.number_input(
                f"Absorbance at λ (A)" + (f" [auto-filled @ {abs_wl_nearest:.1f} nm]" if has_abs else ""),
                value=round(auto_abs_val, 6),
                step=0.001, format="%.5f", key="abs_val"
            )
        with col_b:
            ext_coeff = st.number_input("Extinction coefficient ε (M⁻¹cm⁻¹)", value=43824.0, step=100.0, format="%.1f", key="ext_coeff")
        with col_c:
            path_len = st.number_input("Path length l (cm)", value=1.0, step=0.1, format="%.2f", key="path_len")

        if ext_coeff > 0 and path_len > 0:
            concentration_M  = abs_value / (ext_coeff * path_len)
            concentration_uM = concentration_M * 1e6
        else:
            concentration_M = concentration_uM = 0.0

        st.markdown(f"""
        <div class="stat-row" style="margin-top:0.75rem;">
          <div class="stat">
            <div class="stat-label">Concentration (M)</div>
            <div class="stat-value">{concentration_M:.4e}</div>
          </div>
          <div class="stat">
            <div class="stat-label">Concentration (μM)</div>
            <div class="stat-value">{concentration_uM:.4f} μM</div>
          </div>
          <div class="stat">
            <div class="stat-label">Formula</div>
            <div class="stat-value" style="font-size:0.78rem; color:var(--muted);">c = A / (ε × l)</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.divider()
        st.markdown("#### Aggregation Index")
        st.markdown("""
        <div style="color:var(--muted); font-size:0.82rem; margin-bottom:0.75rem;">
        AggIndex = (Abs₃₅₀ / (Abs₂₈₀ − Abs₃₅₀)) × 100
        </div>
        """, unsafe_allow_html=True)

        # Auto-fill 280 and 350 from parsed absorbance
        if has_abs:
            wl280_n, _ = nearest(abs_series.index.values, 280.0)
            wl350_n, _ = nearest(abs_series.index.values, 350.0)
            auto_280 = float(abs_series.loc[wl280_n]) if not np.isnan(abs_series.loc[wl280_n]) else 0.500
            auto_350 = float(abs_series.loc[wl350_n]) if not np.isnan(abs_series.loc[wl350_n]) else 0.020
            st.markdown(f"""
            <div style="font-size:0.8rem; color:#0a9e7a; margin-bottom:0.5rem;">
            ✅ Auto-filled from file — Abs₂₈₀ @ {wl280_n:.2f} nm, Abs₃₅₀ @ {wl350_n:.2f} nm
            </div>
            """, unsafe_allow_html=True)
        else:
            auto_280, auto_350 = 0.500, 0.020

        col_e, col_f = st.columns(2)
        with col_e:
            abs_280 = st.number_input("Absorbance at 280 nm", value=round(auto_280, 6), step=0.001, format="%.5f", key="abs280")
        with col_f:
            abs_350 = st.number_input("Absorbance at 350 nm", value=round(auto_350, 6), step=0.001, format="%.5f", key="abs350")

        denom = abs_280 - abs_350
        if denom > 0:
            agg_index = (abs_350 / denom) * 100
            agg_color = "#0a9e7a" if agg_index < 5 else ("#e67e22" if agg_index < 15 else "#c0392b")
            agg_label = "Low aggregation" if agg_index < 5 else ("Moderate aggregation" if agg_index < 15 else "High aggregation")
        else:
            agg_index = float("nan")
            agg_color = "#5a6a7e"
            agg_label = "Invalid (Abs₂₈₀ must be > Abs₃₅₀)"

        st.markdown(f"""
        <div class="stat-row" style="margin-top:0.75rem;">
          <div class="stat">
            <div class="stat-label">Aggregation Index</div>
            <div class="stat-value" style="font-size:1.6rem; color:{agg_color};">
              {"%.2f" % agg_index if not np.isnan(agg_index) else "—"} %
            </div>
          </div>
          <div class="stat">
            <div class="stat-label">Interpretation</div>
            <div class="stat-value" style="color:{agg_color}; font-size:0.85rem;">{agg_label}</div>
          </div>
          <div class="stat">
            <div class="stat-label">Abs₂₈₀ − Abs₃₅₀</div>
            <div class="stat-value">{denom:.5f}</div>
          </div>
          <div class="stat">
            <div class="stat-label">Abs₂₈₀</div>
            <div class="stat-value">{abs_280:.5f}</div>
          </div>
          <div class="stat">
            <div class="stat-label">Abs₃₅₀</div>
            <div class="stat-value">{abs_350:.5f}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        if has_abs:
            with st.expander("🔍 Full absorbance spectrum table"):
                abs_df_show = pd.DataFrame({
                    "Wavelength (nm)": abs_series.index,
                    "Absorbance (a.u.)": abs_series.values,
                })
                st.dataframe(abs_df_show, use_container_width=True, height=250)
                csv_abs = abs_df_show.to_csv(index=False).encode("utf-8")
                st.download_button("⬇ Download absorbance spectrum", csv_abs, "absorbance_spectrum.csv", "text/csv", key="dl_abs")

# ── RAW DATA TAB ──────────────────────────────────────────────────────────────
with tab_data:
    st.markdown('<div class="card-title">Raw Intensity Matrix (wavelengths × time)</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        wl_from = st.number_input("Wavelength from (nm)", value=float(wavelengths.min()), format="%.1f")
    with col2:
        wl_to   = st.number_input("Wavelength to (nm)",   value=float(wavelengths.max()), format="%.1f")
    with col3:
        t_stride = st.number_input("Show every N-th time point", value=10, min_value=1, max_value=100)

    wl_mask = (wavelengths >= wl_from) & (wavelengths <= wl_to)
    sub = intensity.loc[wavelengths[wl_mask], times[::int(t_stride)]]
    st.dataframe(sub.style.background_gradient(cmap="Blues"), use_container_width=True, height=400)

    st.divider()

    # ── EXCEL EXPORT ──────────────────────────────────────────────────────────
    st.markdown('<div class="card-title">📥 Export to Excel</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="color:var(--muted); font-size:0.85rem; margin-bottom:1rem;">
    Generates a single <strong>.xlsx</strong> workbook with one sheet per analysis type —
    ready to open directly in Excel.
    </div>
    """, unsafe_allow_html=True)

    export_fname = st.text_input("Output filename", value=f"{st.session_state.active_file.replace('.txt','').replace('.tsv','')}_analysis")

    col_ex1, col_ex2 = st.columns(2)
    with col_ex1:
        include_raw       = st.checkbox("Raw intensity matrix",      value=True)
        include_kinetics  = st.checkbox("Kinetics trace",            value=True)
        include_spectra   = st.checkbox("Spectra snapshot",          value=True)
    with col_ex2:
        include_irif      = st.checkbox("IR/IF ratio over time",     value=True)
        include_pie       = st.checkbox("I350/I330 ratio over time",  value=True)
        include_absorbance= st.checkbox("Absorbance spectrum",        value=ds.get("has_absorbance", False))
        include_summary   = st.checkbox("Summary sheet",             value=True)

    def build_excel():
        buf = io.BytesIO()
        analysis_mask_ex = (times >= t_start) & (times <= t_end)

        with pd.ExcelWriter(buf, engine="openpyxl") as writer:

            # ── Sheet 1: Summary ──────────────────────────────────────────
            if include_summary:
                summary_rows = [
                    ["SpectraKinetics Export", ""],
                    ["File", st.session_state.active_file],
                    ["Export date", pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")],
                    ["", ""],
                    ["=== Dataset Info ===", ""],
                    ["Wavelength range (nm)", f"{wavelengths.min():.2f} – {wavelengths.max():.2f}"],
                    ["Time range (s)", f"{times.min():.2f} – {times.max():.2f}"],
                    ["Number of wavelengths", len(wavelengths)],
                    ["Number of time points", len(times)],
                    ["Excitation wavelength (nm)", 280],
                    ["Absorbance data present", "Yes" if ds.get("has_absorbance") else "No"],
                    ["", ""],
                    ["=== Analysis Parameters ===", ""],
                    ["Kinetics wavelength of interest (nm)", f"{wl_actual:.2f}"],
                    ["Time range analysed (s)", f"{t_start:.2f} – {t_end:.2f}"],
                    ["Spectra time point (s)", f"{t_actual_spec:.2f}"],
                    ["", ""],
                    ["=== IR/IF Ratio ===", ""],
                    ["IR wavelength (nm)", f"{ir_actual:.2f}"],
                    ["IF wavelength (nm)", f"{if_actual:.2f}"],
                    ["Mean IR/IF ratio", f"{mean_irif:.6f}"],
                    ["Final IR/IF ratio", f"{current_irif:.6f}"],
                    ["", ""],
                    ["=== I350/I330 Ratio ===", ""],
                    ["Numerator wavelength (nm)", f"{wl_350_actual:.2f}"],
                    ["Denominator wavelength (nm)", f"{wl_330_actual:.2f}"],
                    ["Mean I350/I330", f"{mean_pie:.6f}"],
                    ["Final I350/I330", f"{current_pie:.6f}"],
                    ["", ""],
                    ["=== Absorbance / Concentration ===", ""],
                    ["Abs at 280 nm", f"{abs_280:.6f}"],
                    ["Abs at 350 nm", f"{abs_350:.6f}"],
                    ["Aggregation Index (%)", f"{'%.4f' % agg_index if not np.isnan(agg_index) else 'N/A'}"],
                    ["Extinction coefficient ε (M⁻¹cm⁻¹)", f"{ext_coeff:.1f}"],
                    ["Path length (cm)", f"{path_len:.2f}"],
                    ["Concentration (M)", f"{concentration_M:.6e}"],
                    ["Concentration (μM)", f"{concentration_uM:.4f}"],
                ]
                summary_df = pd.DataFrame(summary_rows, columns=["Parameter", "Value"])
                summary_df.to_excel(writer, sheet_name="Summary", index=False)
                ws = writer.sheets["Summary"]
                ws.column_dimensions["A"].width = 40
                ws.column_dimensions["B"].width = 30

            # ── Sheet 2: Kinetics trace ───────────────────────────────────
            if include_kinetics:
                kin_mask = (times >= t_start) & (times <= t_end)
                kin_df = pd.DataFrame({
                    "Time (s)": times[kin_mask],
                    f"Intensity @ {wl_actual:.2f} nm (Ex 280 nm)": intensity.loc[wl_actual].values[kin_mask],
                })
                kin_df.to_excel(writer, sheet_name="Kinetics", index=False)
                ws = writer.sheets["Kinetics"]
                ws.column_dimensions["A"].width = 14
                ws.column_dimensions["B"].width = 32

            # ── Sheet 3: Spectra snapshot ─────────────────────────────────
            if include_spectra:
                spec_df_ex = pd.DataFrame({
                    "Wavelength (nm)": wavelengths,
                    f"Intensity @ t={t_actual_spec:.2f} s": intensity[t_actual_spec].values,
                })
                spec_df_ex.to_excel(writer, sheet_name="Spectra", index=False)
                ws = writer.sheets["Spectra"]
                ws.column_dimensions["A"].width = 18
                ws.column_dimensions["B"].width = 28

            # ── Sheet 4: IR/IF ratio ──────────────────────────────────────
            if include_irif:
                t_ir_ex, ir_ex, _ = get_intensity_series(ds, ir_wl, analysis_mask_ex)
                _,       if_ex, _ = get_intensity_series(ds, if_wl, analysis_mask_ex)
                with np.errstate(divide="ignore", invalid="ignore"):
                    ratio_ex = np.where(if_ex != 0, ir_ex / if_ex, np.nan)
                irif_df_ex = pd.DataFrame({
                    "Time (s)": t_ir_ex,
                    f"IR Intensity @ {ir_actual:.2f} nm": ir_ex,
                    f"IF Intensity @ {if_actual:.2f} nm": if_ex,
                    "IR/IF Ratio": ratio_ex,
                })
                irif_df_ex.to_excel(writer, sheet_name="IR-IF Ratio", index=False)
                ws = writer.sheets["IR-IF Ratio"]
                for col, w in zip(["A","B","C","D"], [14, 28, 28, 16]):
                    ws.column_dimensions[col].width = w

            # ── Sheet 5: I350/I330 ratio ──────────────────────────────────
            if include_pie:
                t_350_ex, i350_ex, _ = get_intensity_series(ds, wl_350_input, analysis_mask_ex)
                _,        i330_ex, _ = get_intensity_series(ds, wl_330_input, analysis_mask_ex)
                with np.errstate(divide="ignore", invalid="ignore"):
                    ratio_pie_ex = np.where(i330_ex != 0, i350_ex / i330_ex, np.nan)
                pie_df_ex = pd.DataFrame({
                    "Time (s)": t_350_ex,
                    f"I @ {wl_350_actual:.2f} nm": i350_ex,
                    f"I @ {wl_330_actual:.2f} nm": i330_ex,
                    f"I{wl_350_actual:.0f}/I{wl_330_actual:.0f} Ratio": ratio_pie_ex,
                })
                pie_df_ex.to_excel(writer, sheet_name="I350-I330 Ratio", index=False)
                ws = writer.sheets["I350-I330 Ratio"]
                for col, w in zip(["A","B","C","D"], [14, 24, 24, 20]):
                    ws.column_dimensions[col].width = w

            # ── Sheet 6: Absorbance spectrum ──────────────────────────────
            if include_absorbance and ds.get("has_absorbance"):
                abs_s = ds["absorbance"]
                abs_df_ex = pd.DataFrame({
                    "Wavelength (nm)": abs_s.index,
                    "Absorbance (a.u.)": abs_s.values,
                })
                abs_df_ex.to_excel(writer, sheet_name="Absorbance Spectrum", index=False)
                ws = writer.sheets["Absorbance Spectrum"]
                ws.column_dimensions["A"].width = 20
                ws.column_dimensions["B"].width = 22

            # ── Sheet 7: Raw intensity matrix ─────────────────────────────
            if include_raw:
                raw_export = intensity.copy()
                raw_export.index.name = "Wavelength (nm) \\ Time (s)"
                raw_export.to_excel(writer, sheet_name="Raw Intensity Matrix")

        buf.seek(0)
        return buf.read()

    if st.button("🔨 Build Excel File"):
        with st.spinner("Building workbook…"):
            excel_bytes = build_excel()
        fname = f"{export_fname}.xlsx"
        st.download_button(
            label="⬇ Download Excel Workbook",
            data=excel_bytes,
            file_name=fname,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="dl_excel_final",
        )
        st.success(f"✅ **{fname}** is ready — click above to download.")
