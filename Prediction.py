import os
import warnings
from datetime import datetime, timedelta

import folium
import joblib
import numpy as np
import pandas as pd
import rasterio
import requests
from folium import *
from folium.plugins import MarkerCluster, HeatMap
from pyproj import CRS, Transformer
from rasterio.merge import merge

# Import the central configuration
from config import MINES_CONFIG, GRID_DENSITY, HIGH_RISK_THRESHOLD, MEDIUM_RISK_THRESHOLD

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

# ===================================================================
# --- CONTROL PANEL ---
# ===================================================================
# Set the mode: 'LIVE' or 'SIMULATE'
MODE = 'LIVE'

# --- For SIMULATE Mode ---
# Manually set a high-risk scenario for testing
SIMULATE_RAINFALL_MM = 15.0
SIMULATE_TEMP = 18.0
SIMULATE_VIBRATION = 1.5
SIMULATE_PORE_PRESSURE = 85.0
SIMULATE_DISPLACEMENT_VELOCITY = 0.35


# ===================================================================
# --- ENHANCED FUNCTIONS ---
# ===================================================================

def get_current_mode():
    """Get the current prediction mode."""
    return MODE


def set_simulation_parameters(rainfall=None, temperature=None, vibration=None,
                              pore_pressure=None, displacement=None):
    """Set simulation parameters programmatically."""
    global SIMULATE_RAINFALL_MM, SIMULATE_TEMP, SIMULATE_VIBRATION
    global SIMULATE_PORE_PRESSURE, SIMULATE_DISPLACEMENT_VELOCITY

    if rainfall is not None:
        SIMULATE_RAINFALL_MM = rainfall
    if temperature is not None:
        SIMULATE_TEMP = temperature
    if vibration is not None:
        SIMULATE_VIBRATION = vibration
    if pore_pressure is not None:
        SIMULATE_PORE_PRESSURE = pore_pressure
    if displacement is not None:
        SIMULATE_DISPLACEMENT_VELOCITY = displacement


def get_simulation_parameters():
    """Get current simulation parameters."""
    return {
        'rainfall': SIMULATE_RAINFALL_MM,
        'temperature': SIMULATE_TEMP,
        'vibration': SIMULATE_VIBRATION,
        'pore_pressure': SIMULATE_PORE_PRESSURE,
        'displacement': SIMULATE_DISPLACEMENT_VELOCITY
    }


import traceback


def get_live_environmental_data(lat, lon):
    """
    Fetches all live weather (rain & temp) and seismic data for the last 3 days.
    ENHANCED: Added detailed error logging for diagnostics.
    """
    print("Fetching all live environmental data...")
    # 1. Fetch weather (rain & temp)
    try:
        weather_url = "https://api.open-meteo.com/v1/forecast"
        weather_params = {"latitude": lat, "longitude": lon, "daily": "precipitation_sum,temperature_2m_mean",
                          "timezone": "auto", "past_days": 3}
        response = requests.get(weather_url, params=weather_params, timeout=15)
        response.raise_for_status()
        weather_data = response.json()

        if 'daily' not in weather_data or not weather_data['daily']['time']:
            print("⚠️ Weather API returned success, but the 'daily' data is missing or empty.")
            return None

        weather_df = pd.DataFrame(weather_data['daily']).rename(
            columns={'time': 'date', 'precipitation_sum': 'rainfall_mm'})
        weather_df['date'] = pd.to_datetime(weather_df['date'])
        weather_df = weather_df.fillna(0).infer_objects(copy=False)
        weather_df = weather_df[weather_df['date'] <= pd.to_datetime(datetime.now().date())]
    except requests.exceptions.ConnectionError as e:
        print("\n❌ CRITICAL ERROR: A connection error occurred.")
        print("   This usually means there is no internet connection or a firewall is blocking the request.")
        print(f"   Details: {e}\n")
        return None
    except requests.exceptions.Timeout as e:
        print(f"\n❌ CRITICAL ERROR: The request to the weather API timed out after 15 seconds.")
        print("   This might indicate a slow internet connection.\n")
        return None
    except requests.exceptions.HTTPError as e:
        print(f"\n❌ CRITICAL ERROR: The weather API returned an HTTP error.")
        print(f"   Status Code: {e.response.status_code}")
        print(f"   Response Body: {e.response.text}\n")
        return None
    except Exception as e:
        print("\n❌ CRITICAL ERROR: An unexpected error occurred while fetching weather data.")
        print("   Full error traceback:")
        traceback.print_exc()
        return None

    # Fetch seismic data
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=3)).strftime('%Y-%m-%d')
    seismic_url = "https://earthquake.usgs.gov/fdsnws/event/1/query"
    seismic_params = {'format': 'geojson', 'starttime': start_date, 'endtime': end_date, 'latitude': lat,
                      'longitude': lon, 'maxradiuskm': 250}
    try:
        response = requests.get(seismic_url, params=seismic_params)
        response.raise_for_status()
        seismic_data = response.json()
        earthquakes = [{'date': pd.to_datetime(event['properties']['time'], unit='ms').date(),
                        'peak_vibration': event['properties']['mag']} for event in seismic_data['features']]
        if earthquakes:
            seismic_df = pd.DataFrame(earthquakes)
            daily_max_vibration = seismic_df.groupby('date')['peak_vibration'].max().reset_index()
            daily_max_vibration['date'] = pd.to_datetime(daily_max_vibration['date'])
        else:
            daily_max_vibration = pd.DataFrame(columns=['date', 'peak_vibration'])
    except Exception:
        daily_max_vibration = pd.DataFrame(columns=['date', 'peak_vibration'])

    live_env_data = pd.merge(weather_df, daily_max_vibration, on='date', how='left').fillna(0)
    live_env_data['rainfall_72hr_mm'] = live_env_data['rainfall_mm'].rolling(window=3, min_periods=1).sum()

    # Simulate geotechnical data
    pore_pressure, strain, displacement = [50.0], [0.001], [0.0]
    for i in range(1, len(live_env_data)):
        rain = live_env_data.loc[i, 'rainfall_72hr_mm']
        pressure_change = (rain / 20.0) - (pore_pressure[-1] / 100.0)
        new_pressure = max(50.0, pore_pressure[-1] + pressure_change)
        strain_change = 0.0001 * (new_pressure / 50.0)
        displacement_change = 0.05 * (new_pressure / 50.0) * strain[-1] * 1000
        pore_pressure.append(new_pressure)
        strain.append(strain[-1] + strain_change)
        displacement.append(displacement[-1] + displacement_change)

    live_env_data['pore_pressure'] = pore_pressure
    live_env_data['displacement_velocity_mm_day'] = pd.Series(displacement).diff().fillna(0)
    print("✅ Live data processed.")
    return live_env_data


def mosaic_dem_files(dem_folder):
    """Mosaics all GeoTIFF files in a folder into a single raster object."""
    dem_files = [os.path.join(dem_folder, f) for f in os.listdir(dem_folder) if f.lower().endswith('.tif')]
    if not dem_files:
        raise FileNotFoundError(f"No .tif files found in folder: '{dem_folder}'")

    src_files_to_mosaic = [rasterio.open(fp) for fp in dem_files]
    dest_crs, bounds = src_files_to_mosaic[0].crs, src_files_to_mosaic[0].bounds

    mosaic, out_trans = merge(src_files_to_mosaic)
    for src in src_files_to_mosaic:
        src.close()

    print(f"✅ DEM files mosaicked. Detected CRS: {dest_crs}")
    return mosaic, out_trans, dest_crs, bounds


def create_terrain_dataframe(dem_mosaic, transform, dem_crs, bounds, grid_density):
    """Creates a grid over the DEM and extracts terrain features for each point."""
    print(f"Creating a {grid_density}x{grid_density} grid and extracting terrain features...")
    lons, lats = np.linspace(bounds.left, bounds.right, grid_density), np.linspace(bounds.bottom, bounds.top,
                                                                                   grid_density)
    grid_lon_proj, grid_lat_proj = np.meshgrid(lons, lats)

    transformer_to_wgs84 = Transformer.from_crs(dem_crs, CRS("EPSG:4326"), always_xy=True)
    grid_lon_wgs84, grid_lat_wgs84 = transformer_to_wgs84.transform(grid_lon_proj, grid_lat_proj)

    terrain_features = []
    for lat_proj, lon_proj, lat_wgs84, lon_wgs84 in zip(grid_lat_proj.ravel(), grid_lon_proj.ravel(),
                                                        grid_lat_wgs84.ravel(), grid_lon_wgs84.ravel()):
        col, row = ~transform * (lon_proj, lat_proj)
        col, row = int(col), int(row)
        height, width = dem_mosaic.shape[1], dem_mosaic.shape[2]
        window_size = 21
        half_win = window_size // 2

        if (row > half_win and row < height - half_win and col > half_win and col < width - half_win):
            window = dem_mosaic[0, row - half_win:row + half_win + 1, col - half_win:col + half_win + 1]
            gy, gx = np.gradient(window)
            avg_slope = np.mean(np.degrees(np.arctan(np.sqrt(gx ** 2 + gy ** 2))))
            terrain_features.append({'lat': lat_wgs84, 'lon': lon_wgs84, 'slope': avg_slope})

    print(f"✅ Extracted terrain features for {len(terrain_features)} valid points.")
    return pd.DataFrame(terrain_features)


# ===================================================================
# --- ENHANCED MAIN EXECUTION BLOCK ---
# ===================================================================

def run_prediction_and_generate_map(mine_name="korba_mine", mode=None, simulate_params=None):
    """
    Generates risk maps for a given mine and returns the prediction DataFrame.
    Enhanced with dynamic mode selection and parameter setting.
    """
    global MODE

    # Override mode if specified
    if mode is not None:
        MODE = mode
        print(f"🔄 Mode set to: {MODE}")

    # Set simulation parameters if provided
    if simulate_params is not None and MODE == 'SIMULATE':
        set_simulation_parameters(**simulate_params)
        print(f"🎛️ Simulation parameters updated: {get_simulation_parameters()}")

    result_data = {"dataframe": pd.DataFrame()}
    mine_name_key = mine_name

    for mine_name_loop, config in MINES_CONFIG.items():
        if mine_name_loop != mine_name_key:
            continue

        print(f"\n=================================================")
        print(f"--- Generating maps for: {mine_name.replace('_', ' ').title()} ---")
        print(f"--- Mode: {MODE} ---")

        try:
            # --- 1. Load Data and Model ---
            model_path = f"{mine_name}_model.joblib"
            scaler_path = f"{mine_name}_scaler.joblib"
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            dem_mosaic, transform, dem_crs, bounds = mosaic_dem_files(config['dem_folder'])
            terrain_data = create_terrain_dataframe(dem_mosaic, transform, dem_crs, bounds, GRID_DENSITY)
            if terrain_data.empty:
                print(f"⚠️ Could not extract terrain data for {mine_name}. Skipping.")
                continue

            # --- 2. Get Live Conditions & Predict Risk ---
            live_features_df = terrain_data.copy()

            if MODE == 'LIVE':
                print("\n🌐 --- Running in LIVE Mode ---")
                todays_conditions_df = get_live_environmental_data(config['lat'], config['lon'])
                if todays_conditions_df is None or todays_conditions_df.empty:
                    raise ValueError("Could not fetch live environmental conditions.")

                # Get the last row for the most recent data
                latest_conditions = todays_conditions_df.iloc[-1]

                live_features_df['rainfall_72hr_mm'] = latest_conditions['rainfall_72hr_mm']
                live_features_df['temperature_2m_mean'] = latest_conditions['temperature_2m_mean']
                live_features_df['peak_vibration'] = latest_conditions['peak_vibration']
                live_features_df['pore_pressure'] = latest_conditions['pore_pressure']
                live_features_df['displacement_velocity_mm_day'] = latest_conditions['displacement_velocity_mm_day']

                print(f"📊 Using live data: Rain={latest_conditions['rainfall_72hr_mm']:.1f}mm, "
                      f"Temp={latest_conditions['temperature_2m_mean']:.1f}°C, "
                      f"Vibration={latest_conditions['peak_vibration']:.2f}")

            elif MODE == 'SIMULATE':
                print("\n🔬 --- Running in SIMULATE Mode ---")
                live_features_df['rainfall_72hr_mm'] = SIMULATE_RAINFALL_MM
                live_features_df['temperature_2m_mean'] = SIMULATE_TEMP
                live_features_df['peak_vibration'] = SIMULATE_VIBRATION
                live_features_df['pore_pressure'] = SIMULATE_PORE_PRESSURE
                live_features_df['displacement_velocity_mm_day'] = SIMULATE_DISPLACEMENT_VELOCITY

                print(f"🎛️ Using simulated data: Rain={SIMULATE_RAINFALL_MM}mm, "
                      f"Temp={SIMULATE_TEMP}°C, Vibration={SIMULATE_VIBRATION}, "
                      f"Pressure={SIMULATE_PORE_PRESSURE}kPa, Displacement={SIMULATE_DISPLACEMENT_VELOCITY}mm/day")

            # Predict risk
            features_to_predict = ['slope', 'rainfall_72hr_mm', 'temperature_2m_mean', 'peak_vibration',
                                   'pore_pressure']
            X_grid = live_features_df[features_to_predict]
            X_grid_scaled = scaler.transform(X_grid)
            live_features_df['risk_prob'] = model.predict_proba(X_grid_scaled)[:, 1]
            live_features_df.dropna(subset=['lat', 'lon', 'risk_prob'], inplace=True)

            if live_features_df.empty:
                print(f"⚠️ ERROR: No valid data points remain for {mine_name}.")
                continue

            result_data['dataframe'] = live_features_df
            map_center = [live_features_df['lat'].mean(), live_features_df['lon'].mean()]

            # --- 3. Generate Zoned Risk Map (m_points) ---
            # ------------------- START: ADD THIS NEW BLOCK IN ITS PLACE -------------------

            # --- 3. Generate Zoned Risk Map (Optimized for Performance) ---
            m_points = folium.Map(location=map_center, zoom_start=13, tiles="Esri.WorldImagery")
            at_risk_cluster = MarkerCluster(name="At-Risk Points")

            # First, filter for all at-risk points to get accurate total counts for the summary
            at_risk_df = live_features_df[live_features_df['risk_prob'] >= MEDIUM_RISK_THRESHOLD].copy()
            high_risk_locs = at_risk_df[at_risk_df['risk_prob'] >= HIGH_RISK_THRESHOLD]
            medium_risk_locs = at_risk_df[at_risk_df['risk_prob'] < HIGH_RISK_THRESHOLD]

            # Now, sort by risk and take only the top 250 most critical points to prevent browser crashing
            points_to_display_df = at_risk_df.sort_values(by='risk_prob', ascending=False).head(250)

            print(
                f"INFO: Found {len(at_risk_df)} total at-risk points. Displaying the top {len(points_to_display_df)} on the point map.")

            # Loop ONLY over the top points to create the markers for the map
            for _, row in points_to_display_df.iterrows():
                prob = row['risk_prob']
                popup_html = f"<b>Prob: {prob:.1%}</b> | Slope: {row['slope']:.1f}°<br>Coords: ({row['lat']:.4f}, {row['lon']:.4f})"
                level, color = ("🚨 High Risk", 'red') if prob >= HIGH_RISK_THRESHOLD else ("⚠️ Medium Risk", 'orange')

                folium.CircleMarker(
                    location=(row['lat'], row['lon']), radius=4, color=color, fill=True, fill_color=color,
                    fill_opacity=0.7, popup=folium.Popup(popup_html, max_width=300), tooltip=f"<b>{level}</b>"
                ).add_to(at_risk_cluster)

            m_points.add_child(at_risk_cluster)

            # Add mode indicator to map (This code is the same as before)
            mode_indicator = f'''<div style="position: fixed; top: 10px; right: 10px; z-index:9999; 
                                           background-color: rgba(255, 255, 255, 0.9); border:2px solid #333; 
                                           padding: 8px; border-radius: 5px; font-weight: bold;">
                                           Mode: {MODE}</div>'''

            # The summary box now uses the FULL count of at-risk locations (This code is the same as before)
            summary = f'''<div style="position: fixed; bottom: 50px; left: 50px; width: 220px; z-index:9999; 
                                font-size:16px; background-color: rgba(255, 255, 255, 0.85); border:2px solid black; 
                                padding: 10px; border-radius: 8px;">
                                <b>Risk Summary ({MODE})</b><br><hr style="margin: 5px 0;">
                                <p style="color: red; margin: 0;">🚨 High-Risk: {len(high_risk_locs)}</p>
                                <p style="color: orange; margin: 0;">⚠️ Medium-Risk: {len(medium_risk_locs)}</p></div>'''

            m_points.get_root().html.add_child(folium.Element(mode_indicator))
            m_points.get_root().html.add_child(folium.Element(summary))
            folium.LayerControl().add_to(m_points)
            m_points.save(f"{mine_name}_risk_map_points.html")
            print(f"\n✅ Saved Zoned Risk Map for {mine_name}.")

            # -------------------- END: ADD THIS NEW BLOCK IN ITS PLACE --------------------

            # --- 4. Generate Heatmap (m_heat) ---
            m_heat = folium.Map(location=map_center, zoom_start=13, tiles="Esri.WorldImagery")
            HeatMap(live_features_df[['lat', 'lon', 'risk_prob']].values.tolist(), name="Risk Heatmap").add_to(m_heat)
            m_heat.get_root().html.add_child(folium.Element(mode_indicator))
            m_heat.get_root().html.add_child(folium.Element(summary))
            folium.LayerControl().add_to(m_heat)
            m_heat.save(f"{mine_name}_risk_map_heatmap.html")
            print(f"✅ Saved Heatmap for {mine_name}.")

        except Exception as e:
            print(f"\n❌ An unexpected error occurred while processing {mine_name}: {e}")
            import traceback
            traceback.print_exc()

    return result_data


# In Prediction.py, REPLACE the entire get_forecast_data_and_predict function with this:

def get_forecast_data_and_predict(mine_name):
    """
    Fetches 7-day weather forecast and predicts the max daily risk probability
    using the corrected 5-feature model.
    """
    print(f"\n--- Generating 7-day risk forecast for {mine_name} ---")
    try:
        # --- 1. Load necessary assets ---
        mine_config = MINES_CONFIG[mine_name]
        model = joblib.load(f"{mine_name}_model.joblib")
        scaler = joblib.load(f"{mine_name}_scaler.joblib")
        terrain_df = pd.read_csv(f"{mine_name}_terrain.csv")
        if terrain_df.empty:
            raise ValueError("Terrain data is empty.")

        # --- 2. Fetch 7-day weather forecast ---
        url = "https://api.open-meteo.com/v1/forecast"
        params = {"latitude": mine_config['lat'], "longitude": mine_config['lon'],
                  "daily": "precipitation_sum,temperature_2m_mean",
                  "timezone": "auto", "forecast_days": 7}
        response = requests.get(url, params=params);
        response.raise_for_status()
        forecast_df = pd.DataFrame(response.json()['daily'])
        forecast_df = forecast_df.rename(columns={'time': 'date', 'precipitation_sum': 'rainfall_mm'})
        forecast_df['date'] = pd.to_datetime(forecast_df['date'])

        # --- 3. Prepare data for consistent simulation ---
        live_env_df = get_live_environmental_data(mine_config['lat'], mine_config['lon'])
        if live_env_df is None:
            raise ValueError("Could not get live environmental data for forecast context.")

        # Combine past and future rain for accurate rolling sum
        past_rain = live_env_df[['date', 'rainfall_mm']]
        combined_rain = pd.concat([past_rain, forecast_df[['date', 'rainfall_mm']]]).drop_duplicates(subset=['date'],
                                                                                                     keep='last')
        combined_rain['rainfall_72hr_mm'] = combined_rain['rainfall_mm'].rolling(window=3, min_periods=1).sum()

        # Merge rolling rain back into the forecast dataframe
        forecast_df = pd.merge(forecast_df, combined_rain[['date', 'rainfall_72hr_mm']], on='date')

        # --- 4. Simulate and Predict for each future day ---
        daily_max_risk = []
        features_to_predict = ['slope', 'rainfall_72hr_mm', 'temperature_2m_mean', 'peak_vibration', 'pore_pressure']

        for _, row in forecast_df.iterrows():
            # Create a dataframe for every grid point for this specific day
            day_df = terrain_df.copy()
            day_df['rainfall_72hr_mm'] = row['rainfall_72hr_mm']
            day_df['temperature_2m_mean'] = row['temperature_2m_mean']
            day_df['peak_vibration'] = 0.0  # Assume no earthquakes in forecast

            # Use the SAME stateless formula for pore pressure as the live prediction
            day_df['pore_pressure'] = 50.0 + (day_df['rainfall_72hr_mm'] * 0.75) - (
                        (day_df['temperature_2m_mean'] - 20) * 0.2)
            day_df['pore_pressure'] = day_df['pore_pressure'].clip(lower=40.0)

            # Predict risk using the correct 5 features
            X_day = day_df[features_to_predict]
            predictions = model.predict_proba(scaler.transform(X_day))[:, 1]
            daily_max_risk.append({'date': row['date'], 'max_risk_probability': predictions.max()})

        print("✅ Forecast generation complete.")
        return pd.DataFrame(daily_max_risk)

    except Exception as e:
        print(f"⚠️ An error occurred during forecast generation: {e}")
        traceback.print_exc()
        return pd.DataFrame()


# ===================================================================
# --- UTILITY FUNCTIONS FOR STREAMLIT INTEGRATION ---
# ===================================================================

def get_risk_statistics(prediction_df):
    """Calculate risk statistics from prediction dataframe."""
    if prediction_df.empty:
        return {"high": 0, "medium": 0, "low": 0, "total": 0}

    high_risk = len(prediction_df[prediction_df['risk_prob'] >= HIGH_RISK_THRESHOLD])
    medium_risk = len(prediction_df[(prediction_df['risk_prob'] >= MEDIUM_RISK_THRESHOLD) &
                                    (prediction_df['risk_prob'] < HIGH_RISK_THRESHOLD)])
    low_risk = len(prediction_df[prediction_df['risk_prob'] < MEDIUM_RISK_THRESHOLD])
    total = len(prediction_df)

    return {
        "high": high_risk,
        "medium": medium_risk,
        "low": low_risk,
        "total": total,
        "high_pct": (high_risk / total * 100) if total > 0 else 0,
        "medium_pct": (medium_risk / total * 100) if total > 0 else 0,
        "low_pct": (low_risk / total * 100) if total > 0 else 0
    }


def validate_simulation_parameters(params):
    """Validate simulation parameters are within reasonable ranges."""
    validations = {
        'rainfall': (0, 200, "mm"),
        'temperature': (-20, 60, "°C"),
        'vibration': (0, 10, "magnitude"),
        'pore_pressure': (40, 300, "kPa"),
        'displacement': (0, 5, "mm/day")
    }

    errors = []
    for param, value in params.items():
        if param in validations:
            min_val, max_val, unit = validations[param]
            if not (min_val <= value <= max_val):
                errors.append(f"{param}: {value} {unit} is outside valid range [{min_val}-{max_val}]")

    return errors


if __name__ == '__main__':
    # Example usage
    print("🚀 Running Prediction Script")
    print(f"Current mode: {MODE}")
    if MODE == 'SIMULATE':
        print(f"Simulation parameters: {get_simulation_parameters()}")

    result = run_prediction_and_generate_map()
    if not result['dataframe'].empty:
        stats = get_risk_statistics(result['dataframe'])
        print(f"\n📊 Risk Statistics:")
        print(f"   High Risk: {stats['high']} ({stats['high_pct']:.1f}%)")
        print(f"   Medium Risk: {stats['medium']} ({stats['medium_pct']:.1f}%)")
        print(f"   Low Risk: {stats['low']} ({stats['low_pct']:.1f}%)")
        print(f"   Total Points: {stats['total']}")
    else:
        print("⚠️ No prediction data generated.")