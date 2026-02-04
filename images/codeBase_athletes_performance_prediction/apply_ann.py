# import torch
# import pandas as pd
# import os

# # Paths inside containers
# MODEL_PATH = "/knowledgeBase/model/ann_model.pt"
# DATA_PATH = "/activationBase/activation_data.csv"
# OUTPUT_PATH = "/output/ann_predictions.csv"

# def main():
#     os.makedirs("/output", exist_ok=True)

#     print("Loading ANN model...")
#     model = torch.load(MODEL_PATH, map_location="cpu")
#     model.eval()

#     print("Loading activation data...")
#     X = pd.read_csv(DATA_PATH).values
#     X_tensor = torch.tensor(X, dtype=torch.float32)

#     print("Running inference...")
#     with torch.no_grad():
#         preds = model(X_tensor).numpy()

#     print("Saving output...")
#     pd.DataFrame(preds, columns=["prediction"]).to_csv(OUTPUT_PATH, index=False)

#     print("ANN inference completed successfully.")

# if __name__ == "__main__":
#     main()


import os
import pandas as pd
from tensorflow.keras.models import load_model

# ------------------------------
# Paths inside Docker container
# ------------------------------
MODEL_PATH = "/tmp/knowledgeBase/ann_model.keras"
ACTIVATION_PATH = "/tmp/activationBase/activation_data.csv"
OUTPUT_PATH = "/tmp/results/ann_predictions.csv"

# Ensure output folder exists
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

# ------------------------------
# Load trained ANN model
# ------------------------------
print("Loading ANN model...")
model = load_model(MODEL_PATH)
print("Model loaded successfully.")

# ------------------------------
# Load activation dataset
# ------------------------------
activation_data = pd.read_csv(ACTIVATION_PATH)

# ------------------------------
# Drop target column if it exists
# ------------------------------
if 'Performance_Metric' in activation_data.columns:
    activation_data = activation_data.drop(columns=['Performance_Metric'])
    print("Dropped 'Performance_Metric' column from activation data.")

print("Activation data shape:", activation_data.shape)

# ------------------------------
# Make predictions
# ------------------------------
predictions = model.predict(activation_data)

# ------------------------------
# Save predictions
# ------------------------------
pd.DataFrame(predictions, columns=['prediction']).to_csv(OUTPUT_PATH, index=False)
print(f"Predictions saved to {OUTPUT_PATH}")
