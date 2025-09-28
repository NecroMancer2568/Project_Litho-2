# Save this as inspect_model.py
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from config import MINES_CONFIG

# --- CONFIGURATION ---
# Set the name of the mine you want to inspect
MINE_TO_INSPECT = "korba_mine"
# ---------------------

if __name__ == "__main__":
    try:
        print(f"--- Inspecting model for: {MINE_TO_INSPECT} ---")

        # This is the list of features the model was trained on
        # Make sure it matches the list in your model_training.py file
        features = [
            'slope',
            'rainfall_72hr_mm',
            'temperature_2m_mean',
            'peak_vibration',
            'pore_pressure'
        ]

        # Load the trained model
        model = joblib.load(f"{MINE_TO_INSPECT}_model.joblib")

        # Extract feature importances
        importances = model.feature_importances_

        # Create a pandas DataFrame for easier plotting
        feature_importance_df = pd.DataFrame({
            'feature': features,
            'importance': importances
        }).sort_values('importance', ascending=False)

        print("\nFeature Importances:")
        print(feature_importance_df)

        # Plot the results
        plt.figure(figsize=(10, 6))
        plt.title(f'Feature Importance for {MINE_TO_INSPECT.replace("_", " ").title()}')
        plt.barh(feature_importance_df['feature'], feature_importance_df['importance'])
        plt.xlabel('Importance')
        plt.gca().invert_yaxis()  # Display the most important feature at the top
        plt.tight_layout()

        # Save the plot to a file
        output_filename = f"{MINE_TO_INSPECT}_feature_importance.png"
        plt.savefig(output_filename)
        print(f"\n✅ Saved feature importance plot to: {output_filename}")

    except FileNotFoundError:
        print(f"❌ Error: Model file '{MINE_TO_INSPECT}_model.joblib' not found.")
        print("   Please ensure you have run model_training.py successfully.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")