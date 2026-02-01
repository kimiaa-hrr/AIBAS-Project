import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
import pickle 


# ------------------------------
# Load datasets
# ------------------------------

train_file = "../data/dataset/training_data_running.csv"
val_file   = "../data/dataset/test_data_running.csv"

# Load the datasets
df_train = pd.read_csv(train_file)
df_val   = pd.read_csv(val_file)

# check
print("Training set shape:", df_train.shape)
print("Validation set shape:", df_val.shape)


# ------------------------------
# Separate features and target
# ------------------------------


# All Columns
target = 'Performance_Metric'

# X_train = df_train.drop(columns=[target, 'Sport_Type'])
# y_train = df_train[target]

# print(X_train.dtypes)
# print(y_train.dtypes)

# X_val = df_val.drop(columns=[target, 'Sport_Type'])
# y_val = df_val[target]

# Reducing Columns 
# 'Training_Hours_per_Week','Average_Heart_Rate','Training_Intensity_High','Training_Intensity_Medium','Altitude_Training_High','Altitude_Training_Medium','Sleep_Hours_per_Night','Mental_Focus_Level','Daily_Caloric_Intake','Hydration_Level','Previous_Competition_Performance','Body_Fat_Percentage','Resting_Heart_Rate','VO2_Max','BMI']
specific_features=["Daily_Caloric_Intake","Injury_History_Minor","Injury_History_Major","Training_Intensity_Medium","Training_Intensity_High","BMI"]

# X_train = df_train.drop(columns=[target])
X_train = df_train[specific_features]
y_train = df_train[target]

print(X_train.dtypes)
print(y_train.dtypes)

X_val = df_val[specific_features]
y_val = df_val[target]

# Compute correlation of each column with the target
correlations = X_train.corrwith(y_train).sort_values(key=abs, ascending=False)
print("Columns ranked by correlation with target:")
print(correlations)
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

# with open("../model/OLS/currentOlsSolution.pkl", "wb") as f:
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

# Determine min and max for the line
min_val = min(y_val.min(), y_pred_val.min())
max_val = max(y_val.max(), y_pred_val.max())

plt.figure(figsize=(6,6))
plt.scatter(y_val, y_pred_val, alpha=0.6)
plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--')
plt.xlabel("Actual Performance_Index")
plt.ylabel("Predicted Performance_Index")
plt.title("OLS: Predicted vs Actual")
plt.grid(True)
plt.savefig("../documentation/OLS/pred_vs_actual_OLS_running.png")  # saves the figure
plt.close()
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
plt.savefig("../documentation/OLS/residuals_vs_fitted_OLS_running.png")  # saves the figure
plt.close()