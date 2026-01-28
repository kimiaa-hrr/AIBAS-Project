# ann_model.py
# ANN model with training and saved plots

import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


# --------------------------------------------------
# 1. Load dataset
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "dataset", "training_data.csv")

df = pd.read_csv(DATA_PATH)
print("Dataset loaded successfully")


# --------------------------------------------------
# 2. Target
# --------------------------------------------------
target_column = "Performance_Metric"


# --------------------------------------------------
# 3. Numeric features only
# --------------------------------------------------
numeric_df = df.select_dtypes(include=["number"])
X = numeric_df.drop(columns=[target_column])
y = numeric_df[target_column]


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
    Dense(1)
])


# --------------------------------------------------
# 7. Compile
# --------------------------------------------------
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="mse",
    metrics=["mae"]
)


# --------------------------------------------------
# 8. Train (IMPORTANT: history is defined HERE)
# --------------------------------------------------
history = model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)


# --------------------------------------------------
# 9. Evaluate
# --------------------------------------------------
loss, mae = model.evaluate(X_test, y_test, verbose=0)
print("Training completed successfully ✅")
print(f"Test MAE: {mae:.4f}")


# --------------------------------------------------
# 10. Save plots (NO GUI REQUIRED)
# --------------------------------------------------

# Loss plot
plt.figure()
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss (MSE)")
plt.title("Loss vs Epochs")
plt.legend()
plt.grid(True)
plt.savefig("loss_vs_epochs.png")
plt.close()

# MAE plot
plt.figure()
plt.plot(history.history["mae"], label="Training MAE")
plt.plot(history.history["val_mae"], label="Validation MAE")
plt.xlabel("Epoch")
plt.ylabel("MAE")
plt.title("MAE vs Epochs")
plt.legend()
plt.grid(True)
plt.savefig("mae_vs_epochs.png")
plt.close()

print("Plots saved: loss_vs_epochs.png and mae_vs_epochs.png")
