# Risk Prediction Dashboard

An interactive web-based dashboard for geotechnical risk monitoring and prediction.

## Features

### 🗺️ Real-time Risk Maps
- Interactive map with risk zone visualization
- Color-coded risk levels (High/Medium/Low)
- Clickable markers with detailed information
- Real-time updates from prediction model

### 📊 Probability-based Forecasts
- 24-hour risk probability forecasts
- Environmental condition predictions
- Interactive charts and graphs
- Historical data analysis

### 🚨 Alert Mechanisms
- Real-time alert system
- Configurable risk thresholds
- Multiple alert types (Email, SMS, Sound)
- Severity-based alert categorization

### 📈 Sensor Data Monitoring
- Live geotechnical sensor data
- Environmental factors (rainfall, temperature, vibrations)
- Pore pressure and displacement monitoring
- Historical data visualization

### ⚙️ Configuration Panel
- Adjustable risk thresholds
- Alert preferences
- Data export capabilities
- System settings management

## Quick Start

### Option 1: Using the startup script (Recommended)
```bash
python start_dashboard.py
```

### Option 2: Manual startup
```bash
python app.py
```
Then open http://localhost:5053/api/dashboard in your browser.

## API Endpoints

- `/api/dashboard` - Main dashboard interface
- `/api/risk-map-html` - Generated risk map HTML
- `/api/live-data` - Current sensor data
- `/api/risk-summary` - Risk statistics and alerts
- `/api/forecast-data` - 24-hour forecast data
- `/api/sensor-history` - Historical sensor data
- `/api/risk-zones` - Risk zone coordinates and data
- `/api/alerts` - Current active alerts

## Dashboard Components

### 1. Risk Overview Cards
- High Risk Zones count
- Medium Risk Zones count
- Safe Zones count
- Overall Risk Percentage

### 2. Interactive Risk Map
- Leaflet-based interactive map
- Real-time risk zone markers
- Detailed popup information
- Map view controls (Points/Heatmap)

### 3. Live Sensor Data
- Rainfall (72-hour cumulative)
- Temperature
- Vibration levels
- Pore pressure
- Displacement velocity

### 4. Risk Forecast Chart
- 24-hour risk probability forecast
- Interactive timeline
- Trend analysis

### 5. Alert System
- Real-time alert notifications
- Alert severity levels
- Action recommendations
- Dismissible alerts

### 6. Environmental Conditions
- Multi-parameter environmental chart
- Data export functionality
- Chart view toggles

### 7. Configuration Panel
- Risk threshold adjustment
- Alert preferences
- Settings persistence

## Data Sources

The dashboard integrates with your existing risk prediction model that processes:

- **Geotechnical sensor data**: displacement, strain, pore pressure
- **Environmental factors**: rainfall, temperature, vibrations
- **Digital Elevation Models (DEM)**: terrain analysis

## Customization

### Risk Thresholds
- High Risk: 70% (adjustable 50-90%)
- Medium Risk: 35% (adjustable 20-60%)

### Alert Settings
- Email notifications
- SMS alerts
- Sound alerts
- Configurable thresholds

### Data Export
- CSV export for all chart data
- Historical data download
- Risk zone data export

## Technical Requirements

- Python 3.7+
- Flask
- Required Python packages (see requirements.txt)
- Modern web browser with JavaScript enabled

## Browser Compatibility

- Chrome 80+
- Firefox 75+
- Safari 13+
- Edge 80+

## Mobile Support

The dashboard is fully responsive and works on:
- Tablets (iPad, Android tablets)
- Mobile phones (iOS, Android)
- Touch-enabled devices

## Troubleshooting

### Common Issues

1. **Port already in use**
   - Change port in app.py (line 225)
   - Kill existing process: `lsof -ti:5053 | xargs kill`

2. **Missing dependencies**
   - Install requirements: `pip install -r requirements.txt`

3. **Map not loading**
   - Check internet connection
   - Verify Leaflet CDN access

4. **Data not updating**
   - Check Flask server logs
   - Verify API endpoints are responding

### Performance Tips

- Use Chrome for best performance
- Close unnecessary browser tabs
- Ensure stable internet connection
- Monitor server resources

## Security Notes

- Dashboard runs on localhost by default
- No authentication implemented (add if needed)
- API endpoints are publicly accessible
- Consider adding authentication for production use

## Support

For technical support or feature requests, please check:
1. Flask server logs for errors
2. Browser console for JavaScript errors
3. Network tab for API call failures

## License

This dashboard is part of the Project Litho risk prediction system.
