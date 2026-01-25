import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(BASE_DIR, "..", "data", "dataset", "training_data.csv")

df = pd.read_csv(data_path)
print("Dataset loaded successfully")


# -------------------------------
# Target column
# -------------------------------
# Target column
target_column = "Performance_Metric"

# Automatically select only numeric columns
numeric_df = df.select_dtypes(include=["number"])

X = numeric_df.drop(columns=[target_column])
y = numeric_df[target_column]

print("Features used:", X.columns.tolist())

# -------------------------------
# Train-test split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# Scale data
# -------------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -------------------------------
# Build ANN
# -------------------------------
model = Sequential([
    Dense(64, activation="relu", input_shape=(X_train.shape[1],)),
    Dense(32, activation="relu"),
    Dense(1)
])

# -------------------------------
# Compile model
# -------------------------------
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="mse",
    metrics=["mae"]
)

# -------------------------------
# Train model
# -------------------------------
model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2
)

# -------------------------------
# Evaluate model
# -------------------------------
loss, mae = model.evaluate(X_test, y_test)

print("Training completed successfully ✅")
print("Test MAE:", mae)
