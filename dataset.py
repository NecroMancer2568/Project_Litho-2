# Save this as create_dataset.py
import os, sys, warnings, requests
import pandas as pd
import numpy as np
from datetime import datetime
import rasterio
from rasterio.merge import merge
from pyproj import CRS, Transformer
from skimage.filters import laplace
from config import MINES_CONFIG, START_DATE, END_DATE, GRID_DENSITY, INSTABILITY_THRESHOLD

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)


def mosaic_dem_files(dem_folder):
    dem_files = [os.path.join(dem_folder, f) for f in os.listdir(dem_folder) if f.lower().endswith('.tif')]
    if not dem_files: raise FileNotFoundError(f"No .tif files found in folder: '{dem_folder}'")
    print(f"Found {len(dem_files)} DEM files. Mosaicking...")
    src_files_to_mosaic = [rasterio.open(fp) for fp in dem_files]
    dest_crs, bounds = src_files_to_mosaic[0].crs, src_files_to_mosaic[0].bounds
    mosaic, out_trans = merge(src_files_to_mosaic)
    for src in src_files_to_mosaic: src.close()
    print(f"✅ DEM files mosaicked. Detected CRS: {dest_crs}");
    return mosaic, out_trans, dest_crs, bounds


def create_terrain_dataframe(dem_mosaic, transform, dem_crs, bounds, grid_density):
    print(f"Creating a {grid_density}x{grid_density} grid and extracting terrain features...")
    lons, lats = np.linspace(bounds.left, bounds.right, grid_density), np.linspace(bounds.bottom, bounds.top,
                                                                                   grid_density)
    grid_lon_proj, grid_lat_proj = np.meshgrid(lons, lats)
    transformer_to_wgs84 = Transformer.from_crs(dem_crs, CRS("EPSG:4326"), always_xy=True)
    grid_lon_wgs84, grid_lat_wgs84 = transformer_to_wgs84.transform(grid_lon_proj, grid_lat_proj)
    terrain_features = []
    for lat_proj, lon_proj, lat_wgs84, lon_wgs84 in zip(grid_lat_proj.ravel(), grid_lon_proj.ravel(),
                                                        grid_lat_wgs84.ravel(), grid_lon_wgs84.ravel()):
        col, row = ~transform * (lon_proj, lat_proj);
        col, row = int(col), int(row)
        height, width = dem_mosaic.shape[1], dem_mosaic.shape[2];
        window_size = 21;
        half_win = window_size // 2
        if (row > half_win and row < height - half_win and col > half_win and col < width - half_win):
            window = dem_mosaic[0, row - half_win:row + half_win + 1, col - half_win:col + half_win + 1]
            gy, gx = np.gradient(window);
            avg_slope = np.mean(np.degrees(np.arctan(np.sqrt(gx ** 2 + gy ** 2))))
            location_id = f"loc_{lat_wgs84:.4f}_{lon_wgs84:.4f}"
            terrain_features.append(
                {'location_id': location_id, 'lat': lat_wgs84, 'lon': lon_wgs84, 'slope': avg_slope})
    print(f"✅ Extracted terrain features for {len(terrain_features)} valid points.")
    return pd.DataFrame(terrain_features)


def get_weather_data_openmeteo(lat, lon, start_date, end_date):
    print("Fetching weather data (rain & temp)...")
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {"latitude": lat, "longitude": lon, "start_date": start_date, "end_date": end_date,
              "daily": "precipitation_sum,temperature_2m_mean", "timezone": "auto"}
    try:
        response = requests.get(url, params=params);
        response.raise_for_status()
        data = response.json()
        df = pd.DataFrame(data['daily']).rename(columns={'time': 'date', 'precipitation_sum': 'rainfall_mm'})
        df['date'] = pd.to_datetime(df['date']);
        df = df.fillna(0)
        print("✅ Weather data fetched.");
        return df
    except Exception as e:
        print(f"❌ Failed to fetch weather data: {e}");
        return None


def get_seismic_data_usgs(lat, lon, start_date, end_date, radius_km=250):
    print("Fetching seismic data from USGS...")
    url = "https://earthquake.usgs.gov/fdsnws/event/1/query"
    params = {'format': 'geojson', 'starttime': start_date, 'endtime': end_date, 'latitude': lat, 'longitude': lon,
              'maxradiuskm': radius_km}
    try:
        response = requests.get(url, params=params);
        response.raise_for_status()
        data = response.json()
        earthquakes = [{'date': pd.to_datetime(event['properties']['time'], unit='ms').date(),
                        'peak_vibration': event['properties']['mag']} for event in data['features']]
        if not earthquakes:
            print("✅ No significant seismic events found.");
            return pd.DataFrame(columns=['date', 'peak_vibration'])
        df = pd.DataFrame(earthquakes)
        daily_max_vibration = df.groupby('date')['peak_vibration'].max().reset_index()
        daily_max_vibration['date'] = pd.to_datetime(daily_max_vibration['date'])
        print(f"✅ Found {len(df)} seismic events.");
        return daily_max_vibration
    except Exception as e:
        print(f"❌ Failed to fetch seismic data: {e}");
        return pd.DataFrame(columns=['date', 'peak_vibration'])


# In dataset.py, REPLACE the simulate_geotech_data function with this:

def simulate_geotech_data(environmental_df):
    """
    Simulates geotechnical sensor data using a consistent, direct formula.
    """
    print("Simulating geotechnical sensor data...")
    # This stateless formula ensures consistency between training and live data.
    environmental_df['pore_pressure'] = 50.0 + (environmental_df['rainfall_72hr_mm'] * 0.75) - (
                (environmental_df['temperature_2m_mean'] - 20) * 0.2)
    # Ensure pressure doesn't drop below a baseline
    environmental_df['pore_pressure'] = environmental_df['pore_pressure'].clip(lower=40.0)

    # We remove the unstable displacement simulation entirely.
    geotech_df = environmental_df[['date', 'pore_pressure']].copy()

    print("✅ Geotechnical data simulated.");
    return geotech_df


def create_training_data(terrain_df, full_env_df):
    print("Creating final training dataset with all features...")
    df = terrain_df.merge(full_env_df, how='cross')
    # This new formula makes slope the primary driver of risk.
    slope_factor = df['slope'] / 45.0  # Normalize slope to a 0-1+ range
    rain_factor = df['rainfall_72hr_mm'] / 50.0  # Normalize recent rainfall
    pressure_factor = (df['pore_pressure'] - 50) / 50.0  # Normalize pore pressure above its baseline

    instability_score = slope_factor * (1 + rain_factor + pressure_factor) + (df['peak_vibration'] * 0.5)
    df['rockfall_occurred'] = (instability_score > INSTABILITY_THRESHOLD).astype(int)
    print(f"✅ Training data created. Rockfall events simulated: {df['rockfall_occurred'].sum()}")
    return df


if __name__ == "__main__":
    for mine_name, config in MINES_CONFIG.items():
        print(f"\n--- Processing data for: {mine_name.upper()} ---")
        try:
            dem_folder, lat, lon = config['dem_folder'], config['lat'], config['lon']
            dem_mosaic, transform, dem_crs, bounds = mosaic_dem_files(dem_folder)
            terrain_data = create_terrain_dataframe(dem_mosaic, transform, dem_crs, bounds, GRID_DENSITY)
            if terrain_data.empty: continue

            weather_data = get_weather_data_openmeteo(lat, lon, START_DATE, END_DATE)
            seismic_data = get_seismic_data_usgs(lat, lon, START_DATE, END_DATE)
            if weather_data is not None:
                environmental_data = pd.merge(weather_data, seismic_data, on='date', how='left').fillna(0).infer_objects(copy=False)
                environmental_data['rainfall_72hr_mm'] = environmental_data['rainfall_mm'].rolling(window=3,
                                                                                                   min_periods=1).mean()
                # The simulation function adds the 'pore_pressure' column directly to environmental_data
                simulate_geotech_data(environmental_data)

                # We can now use the modified DataFrame directly without a merge
                final_env_data = environmental_data
                training_df = create_training_data(terrain_data, final_env_data)

                training_df.to_csv(f"{mine_name}_features.csv", index=False)
                terrain_data.to_csv(f"{mine_name}_terrain.csv", index=False)
                print(f"✅ Saved CSV files for {mine_name}")
        except Exception as e:
            print(f"\nAn error occurred while processing {mine_name}: {e}")