# v10, 14 May 2026

# =====================
# SPECTRA ANALYSIS (FINAL CLEAN VERSION)
# =====================
if page == "Spectra Analysis":

    st.title("Spectra Analysis")

    st.subheader("AUC Ratio Settings")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**IR Range (nm)**")
        ir_start = st.number_input("IR Start", value=270.0, key="ir_start")
        ir_end = st.number_input("IR End", value=300.0, key="ir_end")

    with col2:
        st.markdown("**IF Range (nm)**")
        if_start = st.number_input("IF Start", value=320.0, key="if_start")
        if_end = st.number_input("IF End", value=390.0, key="if_end")

    ir_start, ir_end = sorted([ir_start, ir_end])
    if_start, if_end = sorted([if_start, if_end])

    # =====================
    # TABLE (WITH AUC)
    # =====================
    rows = []

    for i, (name, d) in enumerate(data.items()):

        if 280 not in d['spectra']:
            continue

        wl = d['wavelengths']
        y = d['spectra'][280]

        # ---- IR peak ----
        ir_idx = np.argmax(y)
        ir_peak = wl[ir_idx]
        ir_int = y[ir_idx]

        # ---- IF peak ----
        mask_peak = (wl >= 300) & (wl <= 390)

        if np.any(mask_peak):
            y_if = y[mask_peak]
            wl_if = wl[mask_peak]
            idx = np.argmax(y_if)
            if_peak = wl_if[idx]
            if_int = y_if[idx]
        else:
            if_peak = np.nan
            if_int = np.nan

        # =====================
        # ✅ AUC CALCULATIONS
        # =====================
        mask_ir = (wl >= ir_start) & (wl <= ir_end)
        auc_ir = np.trapezoid(y[mask_ir], wl[mask_ir]) if np.any(mask_ir) else np.nan

        mask_if_auc = (wl >= if_start) & (wl <= if_end)
        auc_if = np.trapezoid(y[mask_if_auc], wl[mask_if_auc]) if np.any(mask_if_auc) else np.nan

        irif = auc_ir / auc_if if (not np.isnan(auc_if) and auc_if != 0) else np.nan

        # ---- original ratio preserved ----
        nearest = lambda v: np.argmin(np.abs(wl - v))
        pie = y[nearest(350)] / y[nearest(330)] if y[nearest(330)] != 0 else np.nan

        rows.append({
            "File": name,
            "Index": i,
            "IR/IF": irif,
            "IR/IF (AUC)": irif,
            "AUC IR": auc_ir,
            "AUC IF": auc_if,
            "I350/I330": pie,
            "Aggregation Index": np.nan,
            "Concentration (mg/mL)": np.nan,
            "IR (nm)": ir_peak,
            "IR Peak Intensity": ir_int,
            "IF (nm)": if_peak,
            "IF Peak Intensity": if_int
        })

    df = pd.DataFrame(rows)

    st.dataframe(df, use_container_width=True)

    # =====================
    # ✅ FULL SPECTRA OVERLAY
    # =====================
    st.header(f"Spectra Overlay (Ex {ex_toggle})")

    fig = go.Figure()

    for name, d in data.items():
        if ex_toggle in d['spectra']:
            fig.add_trace(go.Scatter(
                x=d['wavelengths'],
                y=d['spectra'][ex_toggle],
                name=name
            ))

    import uuid

    if len(fig.data) == 0:
        st.warning("No spectra available for selected excitation.")
    else:
        fig.update_layout(
            xaxis_title="Wavelength (nm)",
            yaxis_title="Intensity",
            template="plotly_white"
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            key=f"spectra_overlay_{uuid.uuid4()}"
        )

    # =====================
    # ✅ APIES
    # =====================
    st.header("APIES (All Metrics Overlay)")

    fig2 = go.Figure()

    fig2.add_trace(go.Scatter(
        x=df['Index'],
        y=df['IR/IF'],
        name='IR/IF (AUC)',
        mode='lines+markers'
    ))

    fig2.add_trace(go.Scatter(
        x=df['Index'],
        y=df['I350/I330'],
        name='I350/I330',
        mode='lines+markers'
    ))

    st.plotly_chart(fig2, use_container_width=True, key="apies_plot")


# =====================
# KINETICS (RESTORED)
# =====================
if page == "Kinetics":

    st.title("Kinetics Analysis")

    segments_280, segments_350 = [], []
    sorted_items = sorted(data.items(), key=lambda x: x[0])
    time_offset = 0

    for name, d in sorted_items:
        if d['kinetics'] is None:
            continue

        kin = d['kinetics']
        times = kin['times']

        if len(times) == 0:
            continue

        wl = kin['wavelengths']
        matrix = kin['matrix']

        if len(wl) == 0 or matrix.size == 0:
            continue

        idx_280 = np.argmin(np.abs(wl-280))
        idx_350 = np.argmin(np.abs(wl-350))

        signal_280 = matrix[idx_280,:]
        signal_350 = matrix[idx_350,:]

        shifted_time = times + time_offset

        segments_280.append((shifted_time, signal_280))
        segments_350.append((shifted_time, signal_350))

        time_offset += (times[-1] - times[0])

    if segments_280:
        fig_k = go.Figure()

        for t,y in segments_280:
            fig_k.add_trace(go.Scatter(x=t, y=y, name='280 nm'))
        for t,y in segments_350:
            fig_k.add_trace(go.Scatter(x=t, y=y, name='350 nm'))

        st.plotly_chart(fig_auc, use_container_width=True, key="auc_plot")

# =====================
# ✅ AUC ANALYSIS (FINAL CLEAN VERSION)
# =====================
if page == "AUC Analysis":

    st.title("AUC Analysis")

    # ---- SELECT DATASET ----
    selected_file = st.selectbox(
        "Dataset",
        list(data.keys()),
        key="auc_dataset_select"
    )

    d = data[selected_file]

    # ---- VALIDATION ----
    if ex_toggle not in d['spectra']:
        st.warning("Selected excitation not available.")
        st.stop()

    wl = d['wavelengths']
    y = d['spectra'][ex_toggle]

    if len(wl) == 0:
        st.warning("No wavelength data available.")
        st.stop()

    # ---- LIMITS ----
    min_wl = float(np.min(wl))
    max_wl = float(np.max(wl))

    # =====================
    # ✅ INPUT RANGE
    # =====================
    st.subheader("Select Wavelength Range")

    col1, col2 = st.columns(2)

    with col1:
        start_wl = st.number_input(
            "Start Wavelength (nm)",
            min_value=min_wl,
            max_value=max_wl,
            value=float(min_wl + 20),
            key="start_wl_input"
        )

    with col2:
        end_wl = st.number_input(
            "End Wavelength (nm)",
            min_value=min_wl,
            max_value=max_wl,
            value=float(max_wl - 20),
            key="end_wl_input"
        )

    # ✅ ensure correct ordering
    start_wl, end_wl = sorted([float(start_wl), float(end_wl)])

    # =====================
    # ✅ SINGLE DATASET AUC
    # =====================
    mask = (wl >= start_wl) & (wl <= end_wl)

    if np.any(mask):
        area = np.trapezoid(y[mask], wl[mask])
    else:
        area = 0

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=wl, y=y, name="Full Spectrum"))

    fig.add_trace(go.Scatter(
        x=wl[mask],
        y=y[mask],
        fill='tozeroy',
        name="Selected Region"
    ))

    st.plotly_chart(fig, use_container_width=True, key="auc_single")

    st.metric("AUC", f"{area:.3f}")
    st.info(f"Range: {start_wl:.1f} nm → {end_wl:.1f} nm")

    # =====================
    # ✅ BATCH AUC
    # =====================
    st.markdown("---")

    run_batch = st.button(
        "Calculate AUC for All Datasets",
        key="auc_batch_button"
    )

    if run_batch:

        results = []

        for name, dataset in data.items():

            if ex_toggle not in dataset['spectra']:
                continue

            wl_full = dataset['wavelengths']
            y_full = dataset['spectra'][ex_toggle]

            if len(wl_full) == 0:
                continue

            mask_full = (wl_full >= start_wl) & (wl_full <= end_wl)

            if not np.any(mask_full):
                continue

            auc_val = np.trapezoid(y_full[mask_full], wl_full[mask_full])

            try:
                timestamp = pd.to_datetime(name.split("_")[0])
            except:
                continue

            results.append({
                "time": timestamp,
                "AUC": auc_val,
                "file": name
            })

        if len(results) == 0:
            st.warning("No valid datasets for AUC calculation.")
        else:
            df_auc = pd.DataFrame(results).sort_values("time")

            # ✅ regression
            time_numeric = (df_auc["time"] - df_auc["time"].iloc[0]).dt.total_seconds()
            y_vals = df_auc["AUC"].values

            coeffs = np.polyfit(time_numeric, y_vals, 1)
            fit_line = np.polyval(coeffs, time_numeric)

            ss_res = np.sum((y_vals - fit_line) ** 2)
            ss_tot = np.sum((y_vals - np.mean(y_vals)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else np.nan

            # ✅ plot
            fig_auc = go.Figure()

            fig_auc.add_trace(go.Scatter(
                x=df_auc["time"],
                y=y_vals,
                mode="lines+markers",
                name="AUC"
            ))

            fig_auc.add_trace(go.Scatter(
                x=df_auc["time"],
                y=fit_line,
                mode="lines",
                name=f"Linear Fit (R² = {r2:.4f})",
                line=dict(dash="dash")
            ))

            fig_auc.update_layout(
                title="AUC vs Time",
                xaxis_title="Time",
                yaxis_title="AUC",
                template="plotly_white"
            )

            st.plotly_chart(fig_auc, use_container_width=True, key="auc_batch_plot")

            st.subheader("AUC Results Table")
            st.dataframe(df_auc)

            st.metric("R² (Linear Fit)", f"{r2:.4f}")
