# Save this as train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report
import xgboost as xgb
import joblib
from config import MINES_CONFIG, SMOTE_SAMPLING_STRATEGY

if __name__ == "__main__":
    for mine_name in MINES_CONFIG.keys():
        print(f"\n--- Training final model for: {mine_name.upper()} ---")
        try:
            features_path = f"{mine_name}_features.csv"
            df = pd.read_csv(features_path)

            # The definitive, final feature list
            features = [
                'slope',
                'rainfall_72hr_mm',
                'temperature_2m_mean',
                'peak_vibration',
                'pore_pressure'
            ]
            target = 'rockfall_occurred'

            X = df[features];
            y = df[target]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

            scaler = StandardScaler();
            X_train_scaled = scaler.fit_transform(X_train);
            X_test_scaled = scaler.transform(X_test)
            smote = SMOTE(sampling_strategy=SMOTE_SAMPLING_STRATEGY, random_state=42)
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

            model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', random_state=42)
            model.fit(X_train_resampled, y_train_resampled)

            print(f"\n--- Model Evaluation for {mine_name} ---")
            y_pred = model.predict(X_test_scaled)
            print(classification_report(y_test, y_pred, target_names=['No Rockfall', 'Rockfall']))

            joblib.dump(model, f"{mine_name}_model.joblib");
            joblib.dump(scaler, f"{mine_name}_scaler.joblib")
            print(f"✅ Saved final model and scaler for {mine_name}")
        except Exception as e:
            print(f"\nAn error occurred while training {mine_name}: {e}")