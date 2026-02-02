import pandas as pd
import pickle
import os
import statsmodels.api as sm

# ------------------------------
# Paths inside the container
# ------------------------------
MODEL_PATH = "/tmp/knowledgeBase/currentOlsSolution.pkl"
ACTIVATION_PATH = "/tmp/activationBase/activation_data.csv"
OUTPUT_PATH = "/tmp/results/predictions.csv"

# Ensure output folder exists
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

# ------------------------------
# Load trained OLS model
# ------------------------------
with open(MODEL_PATH, "rb") as f:
    ols_model = pickle.load(f)

# ------------------------------
# Load activation dataset
# ------------------------------
activation_data = pd.read_csv(ACTIVATION_PATH)

# ------------------------------
# Prepare data for prediction
# ------------------------------
# Add constant (intercept) column for Statsmodels
activation_data_sm = sm.add_constant(activation_data, has_constant='add')

# ------------------------------
# Make predictions
# ------------------------------
predictions = ols_model.predict(activation_data_sm)

# ------------------------------
# Save predictions
# ------------------------------
pd.DataFrame(predictions, columns=['prediction']).to_csv(OUTPUT_PATH, index=False)
print(f"Predictions saved to {OUTPUT_PATH}")
