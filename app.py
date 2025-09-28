# Enhanced app.py with mode selection and simulate feature controls

import streamlit as st
import pandas as pd
import folium
import os
import time
from streamlit.components.v1 import html
from Prediction import *
# Import the prediction script's main function and config
from Prediction import run_prediction_and_generate_map, get_live_environmental_data, get_forecast_data_and_predict
from config import MINES_CONFIG, HIGH_RISK_THRESHOLD, MEDIUM_RISK_THRESHOLD

# --- Page Configuration ---
st.set_page_config(
    page_title="Team LITHO",
    page_icon="⛏️",
    layout="wide"
)


# --- Helper Function ---
def read_html_file(path):
    """Reads and returns the content of an HTML file."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return None


def update_prediction_mode_and_params(mode, simulate_params=None):
    """Update prediction mode and parameters using the enhanced Prediction module."""
    try:
        # Import and reload the Prediction module to get fresh functions
        import importlib
        import Prediction
        importlib.reload(Prediction)

        # Use the enhanced functions from Prediction module
        if simulate_params and mode == 'SIMULATE':
            # Validate parameters first
            errors = Prediction.validate_simulation_parameters(simulate_params)
            if errors:
                raise ValueError(f"Invalid parameters: {', '.join(errors)}")

            Prediction.set_simulation_parameters(**simulate_params)

        return True
    except Exception as e:
        print(f"Error updating prediction mode: {e}")
        return False


# --- Main Application ---
st.title("⛏ PROJECT LITHO")
st.markdown("An interactive tool for visualizing real-time rockfall risk based on predictive modeling.")

# Initialize session state
if 'prediction_df' not in st.session_state:
    st.session_state.prediction_df = pd.DataFrame()
if 'forecast_df' not in st.session_state:
    st.session_state.forecast_df = pd.DataFrame()

# Sidebar for controls
with st.sidebar:
    st.header("⚙️ Controls")

    # Mine selection
    mine_name_key = st.selectbox(
        "Select Mine Location",
        options=list(MINES_CONFIG.keys()),
        format_func=lambda x: x.replace('_', ' ').title()
    )

    st.divider()

    # Mode selection
    st.subheader("🔄 Prediction Mode")
    prediction_mode = st.radio(
        "Choose prediction mode:",
        options=['LIVE', 'SIMULATE'],
        index=0,
        help="LIVE: Uses real-time weather and seismic data\nSIMULATE: Uses custom input parameters for testing scenarios"
    )

    # Simulate mode parameters
    simulate_params = {}
    if prediction_mode == 'SIMULATE':
        st.subheader("🎛️ Simulation Parameters")
        st.markdown("*Adjust these values to test different risk scenarios*")

        with st.expander("Environmental Conditions", expanded=True):
            simulate_params['rainfall'] = st.slider(
                "72-Hour Rainfall (mm)",
                min_value=0.0,
                max_value=100.0,
                value=15.0,
                step=1.0,
                help="Total rainfall in the past 72 hours"
            )

            simulate_params['temperature'] = st.slider(
                "Average Temperature (°C)",
                min_value=-10.0,
                max_value=50.0,
                value=18.0,
                step=0.5,
                help="Average daily temperature"
            )

        with st.expander("Seismic & Geotechnical", expanded=True):
            simulate_params['vibration'] = st.slider(
                "Peak Vibration (Magnitude)",
                min_value=0.0,
                max_value=5.0,
                value=1.5,
                step=0.1,
                help="Maximum seismic activity magnitude"
            )

            simulate_params['pore_pressure'] = st.slider(
                "Pore Pressure (kPa)",
                min_value=50.0,
                max_value=200.0,
                value=85.0,
                step=5.0,
                help="Water pressure in rock fractures"
            )

            simulate_params['displacement'] = st.slider(
                "Displacement Velocity (mm/day)",
                min_value=0.0,
                max_value=2.0,
                value=0.35,
                step=0.01,
                help="Rate of rock mass movement"
            )

        # Risk scenario presets
        st.subheader("🎯 Risk Scenario Presets")
        col1, col2 = st.columns(2)

        with col1:
            if st.button("🟢 Low Risk", help="Safe operating conditions"):
                simulate_params = {
                    'rainfall': 5.0,
                    'temperature': 25.0,
                    'vibration': 0.2,
                    'pore_pressure': 55.0,
                    'displacement': 0.05
                }
                st.rerun()

        with col2:
            if st.button("🔴 High Risk", help="Dangerous conditions"):
                simulate_params = {
                    'rainfall': 50.0,
                    'temperature': 5.0,
                    'vibration': 3.0,
                    'pore_pressure': 150.0,
                    'displacement': 1.5
                }
                st.rerun()

    st.divider()

    # Information about the current mode
    if prediction_mode == 'LIVE':
        st.info("🌐 **LIVE Mode**: Using real-time weather and seismic data from external APIs")
    else:
        st.warning("🔬 **SIMULATE Mode**: Using custom parameters for scenario testing")

    st.markdown("---")

    # Refresh button
    if st.button("🔄 REFRESH DATA & GENERATE MAPS", type="primary", use_container_width=True):
        with st.spinner(f"Processing {prediction_mode} mode for {mine_name_key}..."):
            try:
                # Update the prediction mode and parameters
                success = update_prediction_mode_and_params(prediction_mode,
                                                            simulate_params if prediction_mode == 'SIMULATE' else None)

                if not success:
                    st.error("❌ Failed to update prediction parameters.")
                    st.stop()

                # Force reload the Prediction module to pick up changes
                import importlib
                import Prediction

                importlib.reload(Prediction)

                # Run prediction with updated mode and parameters
                prediction_result = Prediction.run_prediction_and_generate_map(
                    mine_name=mine_name_key,
                    mode=prediction_mode,
                    simulate_params=simulate_params if prediction_mode == 'SIMULATE' else None
                )
                st.session_state.prediction_df = prediction_result['dataframe']

                # Clear forecast data to force regeneration
                st.session_state.forecast_df = pd.DataFrame()

                st.success(f"✅ Successfully updated maps using {prediction_mode} mode!")
                time.sleep(1)
            except Exception as e:
                st.error(f"❌ An error occurred: {e}")
                st.warning("Please ensure DEM files are available and model files have been generated.")
                import traceback

                with st.expander("🔍 Error Details"):
                    st.code(traceback.format_exc())

# Check if essential files exist
model_exists = os.path.exists(f"{mine_name_key}_model.joblib")
scaler_exists = os.path.exists(f"{mine_name_key}_scaler.joblib")
points_map_path = f"{mine_name_key}_risk_map_points.html"
heatmap_path = f"{mine_name_key}_risk_map_heatmap.html"

if not model_exists or not scaler_exists:
    st.warning(
        f"⚠️ Model files for '{mine_name_key}' not found. Please run `python dataset.py` and `python model_training.py` from your terminal first.")
else:
    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["🗺️ Real-Time Risk Maps", "📊 Live Data & Risk Summary", "⚡ Quick Actions"])

    with tab1:
        st.header(f"Risk Visualization for: {mine_name_key.replace('_', ' ').title()}")

        # Display current mode info
        mode_color = "🟢" if prediction_mode == "LIVE" else "🟡"
        st.info(f"{mode_color} Currently displaying results from **{prediction_mode}** mode")

        points_html = read_html_file(points_map_path)
        heat_html = read_html_file(heatmap_path)

        if points_html and heat_html:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("📍 Zoned Risk Map")
                st.markdown("Pins indicate specific points of medium (⚠️) or high (🚨) risk.")
                html(points_html, height=600)
            with col2:
                st.subheader("🔥 Risk Heatmap")
                st.markdown("Provides a generalized view of risk concentration across the area.")
                html(heat_html, height=600)
        else:
            st.info("🗺️ Maps not yet generated. Please click the 'REFRESH' button in the sidebar.")

    with tab2:
        st.header("📊 Environmental Data & Risk Analysis")

        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("🌡️ Current Conditions Feeding the Model")

            if prediction_mode == 'LIVE':
                mine_config = MINES_CONFIG[mine_name_key]
                live_data_df = get_live_environmental_data(mine_config['lat'], mine_config['lon'])

                if live_data_df is not None and not live_data_df.empty:
                    latest_live_data = live_data_df.iloc[-1]

                    # Display metrics in a grid
                    metric_col1, metric_col2, metric_col3 = st.columns(3)
                    metric_col4, metric_col5 = st.columns(2)

                    with metric_col1:
                        st.metric("💧 Rainfall (72hr)", f"{latest_live_data['rainfall_72hr_mm']:.2f} mm")
                    with metric_col2:
                        st.metric("🌡️ Avg. Temperature", f"{latest_live_data['temperature_2m_mean']:.1f} °C")
                    with metric_col3:
                        st.metric("📳 Peak Vibration", f"{latest_live_data['peak_vibration']:.2f} mag")
                    with metric_col4:
                        st.metric("💧 Pore Pressure", f"{latest_live_data['pore_pressure']:.2f} kPa")
                    with metric_col5:
                        st.metric("📏 Displacement", f"{latest_live_data['displacement_velocity_mm_day']:.3f} mm/day")
                else:
                    st.warning("⚠️ Could not fetch live environmental data.")
            else:
                # Show simulated parameters
                st.info("🔬 **Simulation Mode** - Using custom parameters:")

                metric_col1, metric_col2, metric_col3 = st.columns(3)
                metric_col4, metric_col5 = st.columns(2)

                with metric_col1:
                    st.metric("💧 Rainfall (72hr)", f"{simulate_params.get('rainfall', 'N/A')} mm")
                with metric_col2:
                    st.metric("🌡️ Avg. Temperature", f"{simulate_params.get('temperature', 'N/A')} °C")
                with metric_col3:
                    st.metric("📳 Peak Vibration", f"{simulate_params.get('vibration', 'N/A')} mag")
                with metric_col4:
                    st.metric("💧 Pore Pressure", f"{simulate_params.get('pore_pressure', 'N/A')} kPa")
                with metric_col5:
                    st.metric("📏 Displacement", f"{simulate_params.get('displacement', 'N/A')} mm/day")

        with col2:
            st.subheader("🎯 Risk Summary")
            df = st.session_state.prediction_df
            if not df.empty:
                high_risk_count = len(df[df['risk_prob'] >= HIGH_RISK_THRESHOLD])
                medium_risk_count = len(
                    df[(df['risk_prob'] >= MEDIUM_RISK_THRESHOLD) & (df['risk_prob'] < HIGH_RISK_THRESHOLD)])
                low_risk_count = len(df[df['risk_prob'] < MEDIUM_RISK_THRESHOLD])

                st.metric("🚨 High Risk Points", high_risk_count)
                st.metric("⚠️ Medium Risk Points", medium_risk_count)
                st.metric("✅ Low Risk Points", low_risk_count)

                # Risk level visualization
                total_points = len(df)
                if total_points > 0:
                    high_pct = (high_risk_count / total_points) * 100
                    medium_pct = (medium_risk_count / total_points) * 100
                    low_pct = (low_risk_count / total_points) * 100

                    st.markdown("**Risk Distribution:**")
                    st.progress(high_pct / 100, text=f"High Risk: {high_pct:.1f}%")
                    st.progress(medium_pct / 100, text=f"Medium Risk: {medium_pct:.1f}%")
                    st.progress(low_pct / 100, text=f"Low Risk: {low_pct:.1f}%")

        # Risk point coordinates
        st.subheader("📍 Coordinates of At-Risk Points")
        df = st.session_state.prediction_df
        if not df.empty:
            high_risk_df = df[df['risk_prob'] >= HIGH_RISK_THRESHOLD].copy()
            medium_risk_df = df[
                (df['risk_prob'] >= MEDIUM_RISK_THRESHOLD) & (df['risk_prob'] < HIGH_RISK_THRESHOLD)].copy()

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### 🚨 High-Risk Locations")
                if not high_risk_df.empty:
                    st.dataframe(
                        high_risk_df[['lat', 'lon', 'risk_prob', 'slope']].style.format({
                            'risk_prob': '{:.2%}',
                            'lat': '{:.4f}',
                            'lon': '{:.4f}',
                            'slope': '{:.1f}°'
                        }),
                        use_container_width=True
                    )
                else:
                    st.success("✅ No high-risk points detected.")

            with col2:
                st.markdown("#### ⚠️ Medium-Risk Locations")
                if not medium_risk_df.empty:
                    st.dataframe(
                        medium_risk_df[['lat', 'lon', 'risk_prob', 'slope']].style.format({
                            'risk_prob': '{:.2%}',
                            'lat': '{:.4f}',
                            'lon': '{:.4f}',
                            'slope': '{:.1f}°'
                        }),
                        use_container_width=True
                    )
                else:
                    st.success("✅ No medium-risk points detected.")

            # 7-Day Risk Forecast
            st.subheader("📈 7-Day Risk Probability Forecast")
            st.markdown(
                "This chart shows the predicted _maximum_ risk level across the entire site for the coming week.")

            if not st.session_state.prediction_df.empty:
                if st.session_state.forecast_df.empty:
                    with st.spinner("Generating 7-day risk forecast..."):
                        st.session_state.forecast_df = get_forecast_data_and_predict(mine_name_key)

                forecast_df = st.session_state.forecast_df
                if not forecast_df.empty:
                    st.line_chart(
                        forecast_df,
                        x='date',
                        y='max_risk_probability',
                        color="#ff6347"
                    )
                else:
                    st.warning("⚠️ Could not generate risk forecast data.")
            else:
                st.info("📊 Click the 'REFRESH' button in the sidebar to generate the risk forecast.")

        else:
            st.info("📊 No prediction data available. Click the 'REFRESH' button to generate predictions.")

    with tab3:
        st.header("⚡ Quick Actions & Tools")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🔧 Mode Management")
            current_mode = prediction_mode
            st.info(f"Current Mode: **{current_mode}**")


        with col2:
            st.subheader("📋 System Status")
            model_status = "✅ Ready" if model_exists else "❌ Missing"
            scaler_status = "✅ Ready" if scaler_exists else "❌ Missing"
            maps_status = "✅ Generated" if (
                        read_html_file(points_map_path) and read_html_file(heatmap_path)) else "⚠️ Not Generated"

            st.write(f"**Model:** {model_status}")
            st.write(f"**Scaler:** {scaler_status}")
            st.write(f"**Maps:** {maps_status}")

        # Export options
        st.subheader("💾 Export Data")
        if not st.session_state.prediction_df.empty:
            col1, col2 = st.columns(2)

            with col1:
                csv_data = st.session_state.prediction_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Risk Data (CSV)",
                    data=csv_data,
                    file_name=f"{mine_name_key}_risk_analysis_{prediction_mode.lower()}.csv",
                    mime="text/csv"
                )

            with col2:
                if not st.session_state.forecast_df.empty:
                    forecast_csv = st.session_state.forecast_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Forecast (CSV)",
                        data=forecast_csv,
                        file_name=f"{mine_name_key}_forecast.csv",
                        mime="text/csv"
                    )
        else:
            st.info("No data available for export. Generate predictions first.")