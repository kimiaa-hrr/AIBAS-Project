# ANN model with training, evaluation, plots, and activation prediction

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


# --------------------------------------------------
# 1. Paths
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TRAINING_PATH = os.path.join(BASE_DIR, "..", "data", "dataset", "training_data_tennis.csv")
ACTIVATION_PATH = os.path.join(BASE_DIR, "..", "data", "dataset", "activation_data.csv")

print("==================== TRAINING_PATH ==================== ",TRAINING_PATH)
# --------------------------------------------------
# 2. Load dataset
# --------------------------------------------------
df = pd.read_csv(TRAINING_PATH)
print("Dataset loaded successfully")

target_column = "Performance_Metric"


# --------------------------------------------------
# 3. Use numeric features only
# --------------------------------------------------
numeric_df = df.select_dtypes(include=["number"])

X = numeric_df.drop(columns=[target_column])
y = numeric_df[target_column]

# SAVE feature names (IMPORTANT)
feature_columns = X.columns


# --------------------------------------------------
# 4. Train-test split
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# --------------------------------------------------
# 5. Scaling
# --------------------------------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# --------------------------------------------------
# 6. Build ANN
# --------------------------------------------------
model = Sequential([
    Dense(64, activation="relu", input_shape=(X_train.shape[1],)),
    Dense(32, activation="relu"),
    Dense(1, activation="linear")
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="mse",
    metrics=["mae"]
)


# --------------------------------------------------
# 7. Train model
# --------------------------------------------------
history = model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    epochs=50,
    batch_size=32,
    verbose=1
)


# --------------------------------------------------
# 8. Evaluate on test data
# --------------------------------------------------
test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
print("Training completed successfully ✅")
print(f"Test MAE: {test_mae}")

# --------------------------------------------------
# Save trained model
# --------------------------------------------------
MODEL_PATH = os.path.join(BASE_DIR,"", "ann_model.keras")
model.save(MODEL_PATH)
print(f"Model saved at: {MODEL_PATH}")



# --------------------------------------------------
# 9. Plot & save training curves
# --------------------------------------------------
plt.figure()
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss (MSE)")
plt.title("Loss vs Epochs")
plt.legend()
plt.tight_layout()
plt.savefig("loss_vs_epochs.png")
plt.close()

plt.figure()
plt.plot(history.history["mae"], label="Training MAE")
plt.plot(history.history["val_mae"], label="Validation MAE")
plt.xlabel("Epoch")
plt.ylabel("MAE")
plt.title("MAE vs Epochs")
plt.legend()
plt.tight_layout()
plt.savefig("mae_vs_epochs.png")
plt.close()


# --------------------------------------------------
# 10. Activation data prediction (FIXED)
# --------------------------------------------------
activation_df = pd.read_csv(ACTIVATION_PATH)

activation_numeric = activation_df.select_dtypes(include=["number"])

# FORCE same columns as training
activation_numeric = activation_numeric.reindex(
    columns=feature_columns,
    fill_value=0
)

activation_scaled = scaler.transform(activation_numeric)
activation_prediction = model.predict(activation_scaled)

print("Activation prediction:", activation_prediction[0][0])
