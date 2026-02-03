import tensorflow as tf
import pandas as pd
import os

MODEL_PATH = "/knowledgeBase/model/ann_model.keras"
DATA_PATH = "/activationBase/activation_data.csv"
OUTPUT_PATH = "/output/ann_predictions.csv"

def main():
    os.makedirs("/output", exist_ok=True)

    print("Loading ANN model (tf.keras)...")
    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    print("Loading activation data...")
    X = pd.read_csv(DATA_PATH)

    print("Running inference...")
    preds = model.predict(X)

    print("Saving output...")
    pd.DataFrame(preds, columns=["prediction"]).to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("ANN inference completed successfully.")

if __name__ == "__main__":
    main()
