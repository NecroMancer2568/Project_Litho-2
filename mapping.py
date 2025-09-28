import folium
from folium import plugins
from config import HIGH_RISK_THRESHOLD, MEDIUM_RISK_THRESHOLD


def generate_risk_maps(prediction_data, mine_name):
    """
    Generate risk visualization maps (points and heatmap) from prediction data.

    Args:
        prediction_data (dict): Dictionary containing the prediction DataFrame
        mine_name (str): Name of the mine for file naming

    Returns:
        dict: Dictionary containing the DataFrame used for mapping
    """
    df = prediction_data['dataframe']

    # Calculate center coordinates for the map
    center_lat = df['lat'].mean()
    center_lon = df['lon'].mean()

    # Create base maps
    points_map = folium.Map(location=[center_lat, center_lon], zoom_start=14)
    heat_map = folium.Map(location=[center_lat, center_lon], zoom_start=14)

    # Add points layer with risk indicators
    for _, row in df.iterrows():
        if row['risk_prob'] >= HIGH_RISK_THRESHOLD:
            color = 'red'
            icon = '🚨'
        elif row['risk_prob'] >= MEDIUM_RISK_THRESHOLD:
            color = 'orange'
            icon = '⚠️'
        else:
            continue  # Skip low-risk points

        folium.CircleMarker(
            location=[row['lat'], row['lon']],
            radius=8,
            color=color,
            fill=True,
            popup=f"{icon} Risk: {row['risk_prob']:.1%}<br>Slope: {row['slope']:.1f}°",
        ).add_to(points_map)

    # Add heatmap layer
    heat_data = df[['lat', 'lon', 'risk_prob']].values.tolist()
    plugins.HeatMap(heat_data).add_to(heat_map)

    # Save maps
    points_map.save(f"{mine_name}_risk_map_points.html")
    heat_map.save(f"{mine_name}_risk_map_heatmap.html")

    return {'dataframe': df}