import torch
import pandas as pd
import os

# Paths inside containers
MODEL_PATH = "/knowledgeBase/model/ann_model.pt"
DATA_PATH = "/activationBase/activation_data.csv"
OUTPUT_PATH = "/output/ann_predictions.csv"

def main():
    os.makedirs("/output", exist_ok=True)

    print("Loading ANN model...")
    model = torch.load(MODEL_PATH, map_location="cpu")
    model.eval()

    print("Loading activation data...")
    X = pd.read_csv(DATA_PATH).values
    X_tensor = torch.tensor(X, dtype=torch.float32)

    print("Running inference...")
    with torch.no_grad():
        preds = model(X_tensor).numpy()

    print("Saving output...")
    pd.DataFrame(preds, columns=["prediction"]).to_csv(OUTPUT_PATH, index=False)

    print("ANN inference completed successfully.")

if __name__ == "__main__":
    main()
