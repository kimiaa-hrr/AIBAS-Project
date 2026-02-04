import pandas as pd
import pickle
import statsmodels.api as sm
import os

# ------------------------------
# Paths inside the container
# ------------------------------
MODEL_PATH = "/tmp/knowledgeBase/currentOlsSolution.pkl"
ACTIVATION_PATH = "/tmp/activationBase/activation_data.csv"
OUTPUT_PATH = "/tmp/results/predictions.csv"

# ------------------------------
# Ensure output folder exists
# ------------------------------
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

print(f"Activation data shape: {activation_data.shape}")
print(f"Activation columns: {activation_data.columns.tolist()}")

# ------------------------------
# Prepare data for prediction
# ------------------------------
model_columns = ols_model.model.exog_names  # Columns the model expects

# Check if model has a constant
if "const" in model_columns and "const" not in activation_data.columns:
    activation_data = sm.add_constant(activation_data, has_constant='add')
    print("Added constant column to activation data.")

# Add missing columns with zeros
for col in model_columns:
    if col not in activation_data.columns:
        activation_data[col] = 0
        print(f"Added missing column '{col}' with zeros.")

# Reorder columns to match the model
activation_data_aligned = activation_data[model_columns]

print(f"Aligned activation data shape: {activation_data_aligned.shape}")
print(f"Columns after alignment: {activation_data_aligned.columns.tolist()}")

# ------------------------------
# Make predictions
# ------------------------------
predictions = ols_model.predict(activation_data_aligned)

print(f"Predictions preview:\n{predictions[:5]}")

# ------------------------------
# Save predictions
# ------------------------------
pd.DataFrame(predictions, columns=['prediction']).to_csv(OUTPUT_PATH, index=False)
print(f"Predictions saved to {OUTPUT_PATH}")
