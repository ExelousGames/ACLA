import streamlit as st
import numpy as np
import plotly.express as px

def render_feature_calculator(df, form_start, form_end, numeric_cols, selected_option):
    """
    Renders the Feature Change Calculator section.
    """
    with st.expander("Feature Change Calculator"):
        f_col1, f_col2 = st.columns([1, 2])
        with f_col1:
            # Default to speed or gas if available
            default_calc_idx = 0
            if "speed_kmh" in numeric_cols:
                default_calc_idx = numeric_cols.index("speed_kmh")
            
            calc_feature = st.selectbox(
                "Select Feature", 
                numeric_cols, 
                index=default_calc_idx,
                key=f"detailed_calc_feat_{selected_option}"
            )
        
        with f_col2:
            if calc_feature and form_start < form_end and int(form_end) < len(df):
                # Calculate changes
                calc_slice = df.iloc[int(form_start):int(form_end)+1][calc_feature]
                
                # Comprehensive Statistical Analysis
                min_val = calc_slice.min()
                max_val = calc_slice.max()
                mean_val = calc_slice.mean()
                median_val = calc_slice.median()
                std_val = calc_slice.std()
                var_val = calc_slice.var()
                
                # Derivative Stats (Rate of Change)
                diffs = calc_slice.diff().dropna()
                max_rate = diffs.max() if not diffs.empty else 0
                min_rate = diffs.min() if not diffs.empty else 0
                avg_abs_rate = diffs.abs().mean() if not diffs.empty else 0
                
                # Integral (Area under curve approximation)
                area = np.trapz(calc_slice.values)
                
                # Total Change
                total_change = calc_slice.iloc[-1] - calc_slice.iloc[0]

                st.markdown("##### Statistical Analysis")
                
                # Row 1: Range & Central Tendency
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Minimum", f"{min_val:.2f}")
                c2.metric("Maximum", f"{max_val:.2f}")
                c3.metric("Mean", f"{mean_val:.2f}")
                c4.metric("Median", f"{median_val:.2f}")

                # Row 2: Variability & Dynamics
                c5, c6, c7, c8 = st.columns(4)
                c5.metric("Std Dev", f"{std_val:.2f}")
                c6.metric("Max Rate (Δ)", f"{max_rate:.2f}")
                c7.metric("Min Rate (Δ)", f"{min_rate:.2f}")
                c8.metric("Avg Volatility", f"{avg_abs_rate:.2f}", help="Average absolute change between consecutive points")
                
                # Row 3: Cumulative
                c9, c10, c11, c12 = st.columns(4)
                c9.metric("Integral (Area)", f"{area:.2f}", help="Area under the curve (Trapezoidal rule)")
                c10.metric("Sum", f"{calc_slice.sum():.2f}")
                c11.metric("Variance", f"{var_val:.2f}")
                c12.metric("Total Change", f"{total_change:.2f}", help="Difference between end and start value")

                st.markdown("##### Rate of Change Over Time")
                if not diffs.empty:
                    scr_col1, scr_col2 = st.columns([1, 1])
                    with scr_col1:
                         # Smoothing Control
                         smooth_window = st.slider(
                             "Smoothing (Moving Average)", 
                             min_value=1, 
                             max_value=max(2, min(50, len(diffs))), 
                             value=1, 
                             key=f"detailed_roc_smooth_{selected_option}"
                         )

                    data_to_plot = diffs
                    if smooth_window > 1:
                        data_to_plot = diffs.rolling(window=smooth_window, center=True).mean()

                    fig_roc = px.line(
                        x=data_to_plot.index, 
                        y=data_to_plot.values, 
                        labels={'x': 'Index', 'y': f'Change'}, 
                        title=f"Rate of Change (Δ) - {calc_feature} (Window: {smooth_window})"
                    )
                    st.plotly_chart(fig_roc, width='stretch')
