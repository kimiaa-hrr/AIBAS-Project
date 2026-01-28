import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
import pickle 


# ------------------------------
# Load datasets
# ------------------------------

train_file = "../data/dataset/training_data.csv"
val_file   = "../data/dataset/test_data.csv"

# Load the datasets
df_train = pd.read_csv(train_file)
df_val   = pd.read_csv(val_file)

# check
print("Training set shape:", df_train.shape)
print("Validation set shape:", df_val.shape)


# ------------------------------
# Separate features and target
# ------------------------------

target = "Performance_Metric"

X_train = df_train.drop(columns=[target])
y_train = df_train[target]

print(X_train.dtypes)
print(y_train.dtypes)

X_val = df_val.drop(columns=[target])
y_val = df_val[target]

# ------------------------------
# Add constant (intercept) for Statsmodels
# ------------------------------

X_train_sm = sm.add_constant(X_train)
X_val_sm   = sm.add_constant(X_val)

# ------------------------------
# Train OLS model
# ------------------------------

ols_model = sm.OLS(y_train, X_train_sm).fit()
print(ols_model.summary())


# ------------------------------
# Save trained model
# ------------------------------

# with open("currentOlsSolution.pkl", "wb") as f:
#     pickle.dump(ols_model, f)

# ------------------------------
# Make predictions on test set
# ------------------------------

y_pred_val = ols_model.predict(X_val_sm)
# ------------------------------
# Performance metrics
# ------------------------------

r2 = r2_score(y_val, y_pred_val)
rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))

print(f"\nValidation R²: {r2:.3f}")
print(f"Validation RMSE: {rmse:.3f}")

# ------------------------------
# Scatter plot (Predicted vs Actual)
# ------------------------------

plt.figure(figsize=(6,6))
plt.scatter(y_val, y_pred_val, alpha=0.6)
plt.plot([0,1], [0,1], color='red', linestyle='--')  # perfect prediction line
plt.xlabel("Actual Performance_Index")
plt.ylabel("Predicted Performance_Index")
plt.title("OLS: Predicted vs Actual")
plt.grid(True)
plt.savefig("../documentation/OLS/pred_vs_actual_OLS.png")  # saves the figure
plt.show()

# ------------------------------
# Residuals vs Fitted plot
# ------------------------------

residuals = y_train - ols_model.fittedvalues

plt.figure(figsize=(6,4))
plt.scatter(ols_model.fittedvalues, residuals, alpha=0.6)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Fitted Values")
plt.ylabel("Residuals")
plt.title("Residuals vs Fitted")
plt.grid(True)
plt.savefig("../documentation/OLS/residuals_vs_fitted_OLS.png")  # saves the figure
plt.show()