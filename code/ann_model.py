# ============================================================
# ANN Model – Final Version (No Extra Split)
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout
# ============================================================
# 1. Paths
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TRAIN_PATH = os.path.join(BASE_DIR, "..", "data", "dataset", "training_data.csv")
TEST_PATH = os.path.join(BASE_DIR, "..", "data", "dataset", "test_data.csv")

DOC_DIR = os.path.join(BASE_DIR, "..", "documentation", "Ann")
MODEL_DIR = os.path.join(BASE_DIR, "..", "model")

os.makedirs(DOC_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ============================================================
# 2. Load datasets (already split)
# ============================================================

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

print("Training & test datasets loaded")

TARGET = "Performance_Metric"

# ============================================================
# 3. Use numeric features only
# ============================================================

X_train = train_df.select_dtypes(include=["number"]).drop(columns=[TARGET])
y_train = train_df[TARGET]

X_test = test_df.select_dtypes(include=["number"]).drop(columns=[TARGET])
y_test = test_df[TARGET]

# Save feature order (IMPORTANT)
feature_columns = X_train.columns

# ============================================================
# 4. Scaling
# ============================================================

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ============================================================
# 5. Build ANN
# ============================================================

model = Sequential([
    Dense(64, activation="relu", input_shape=(X_train.shape[1],)),
    Dropout(0.2),  # Randomly shuts off 20% of neurons during training
    Dense(32, activation="relu"),
    Dropout(0.2),  # Helps prevent the model from "memorizing" noise
    Dense(1, activation="sigmoid")   # performance normalized 0–1
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="mse",
    metrics=["mae"]
)

# ============================================================
# 6. Train
# ============================================================
# Define the early stopping monitor
early_stopping = EarlyStopping(
    monitor='val_loss', 
    patience=5,           # Number of epochs to wait for improvement
    restore_best_weights=True  # Returns the model to its best version
)

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=32,
    verbose=1,
    callbacks=[early_stopping] # <--- Add this here
)

# ============================================================
# 7. Save training curves
# ============================================================

plt.figure()
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.legend()
plt.title("Loss vs Epochs")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.savefig(os.path.join(DOC_DIR, "loss_vs_epochs.png"))
plt.close()

plt.figure()
plt.plot(history.history["mae"], label="Train MAE")
plt.plot(history.history["val_mae"], label="Val MAE")
plt.legend()
plt.title("MAE vs Epochs")
plt.xlabel("Epoch")
plt.ylabel("MAE")
plt.savefig(os.path.join(DOC_DIR, "mae_vs_epochs.png"))
plt.close()

# ============================================================
# 8. Evaluate
# ============================================================

test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
print(f"Test MAE: {test_mae}")

# --------------------------------------------------
# Save trained model
# --------------------------------------------------
MODEL_PATH = os.path.join(BASE_DIR, "ann_model.keras")
model.save(MODEL_PATH)
print(f"Model saved at: {MODEL_PATH}")

# ============================================================
# 10. ANN Diagnostic Plots (Instructor Requested)
# ============================================================

y_pred = model.predict(X_test).flatten()
residuals = y_test.values - y_pred
std_residuals = residuals / np.std(residuals)

# --- Residuals vs Fitted
plt.figure()
plt.scatter(y_pred, residuals, alpha=0.6)
plt.axhline(0, color="red", linestyle="--")
plt.xlabel("Fitted Values")
plt.ylabel("Residuals")
plt.title("Residuals vs Fitted (ANN)")
plt.savefig(os.path.join(DOC_DIR, "ann_residuals_vs_fitted.png"))
plt.close()

# --- Q-Q Plot
plt.figure()
stats.probplot(residuals, dist="norm", plot=plt)
plt.title("Normal Q-Q Plot (ANN Residuals)")
plt.savefig(os.path.join(DOC_DIR, "ann_qq_plot.png"))
plt.close()

# --- Scale-Location
plt.figure()
plt.scatter(y_pred, np.sqrt(np.abs(std_residuals)), alpha=0.6)
plt.xlabel("Fitted Values")
plt.ylabel("√|Standardized Residuals|")
plt.title("Scale-Location Plot (ANN)")
plt.savefig(os.path.join(DOC_DIR, "ann_scale_location.png"))
plt.close()

# --- Residuals vs Leverage (ANN Approximation)
leverage = np.sum(X_test ** 2, axis=1)
leverage = leverage / leverage.max()

plt.figure()
plt.scatter(leverage, std_residuals, alpha=0.6)
plt.axhline(0, color="red", linestyle="--")
plt.xlabel("Approx. Leverage")
plt.ylabel("Standardized Residuals")
plt.title("Residuals vs Leverage (ANN Approx.)")
plt.savefig(os.path.join(DOC_DIR, "ann_residuals_vs_leverage.png"))
plt.close()

print("All ANN diagnostics generated successfully")
