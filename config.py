# Save this as config.py

# --- GEOGRAPHIC AND TIME SETTINGS ---
# Add a new entry for each mine you want to model.
# The key (e.g., "korba_mine") will be used for filenames.
MINES_CONFIG = {
    "korba_mine": {
        "dem_folder": "/Users/persistant_brat/DEM_FILES",
        "lat": 22.34,
        "lon": 82.56
    },
    "dipka_mine":{
        "dem_folder": "/Users/persistant_brat/Downloads/dipka",
        "lat": 22.279,
        "lon": 82.209
    },
    "neyveli_mine":{
        "dem_folder": "/Users/persistant_brat/Downloads/neyveli",
        "lat": 11.6,
        "lon": 79.4
    }

































}

# Date range for creating the historical training dataset
START_DATE = '2020-01-01'
END_DATE = '2024-12-31'

# --- MODEL TRAINING TUNING ---
# Adjust SMOTE sensitivity. 1.0 = 50/50 balance. 0.5 = 2:1 balance.
SMOTE_SAMPLING_STRATEGY = 1.0
# Adjust the simulation's sensitivity to create more/fewer rockfall events.
INSTABILITY_THRESHOLD = 1.00

# --- MAP & ALERT TUNING (Probability Threshold Control) ---
HIGH_RISK_THRESHOLD = 0.85
MEDIUM_RISK_THRESHOLD = 0.45
GRID_DENSITY = 100