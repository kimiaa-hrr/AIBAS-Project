import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
import pickle 
from scipy import stats


# ------------------------------
# Load datasets
# ------------------------------

train_file = "../data/dataset/training_data_tennis.csv"
val_file   = "../data/dataset/test_data_tennis.csv"

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
# specific_features=["Daily_Caloric_Intake","Injury_History_Minor","Injury_History_Major","Training_Intensity_Medium","Training_Intensity_High","BMI"]

X_train = df_train.drop(columns=[target])
# X_train = df_train[specific_features]
y_train = df_train[target]

print(X_train.dtypes)
print(y_train.dtypes)

X_val = df_val.drop(columns=[target])
# X_val = df_val[specific_features]
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

with open("../model/OLS/currentOlsSolution.pkl", "wb") as f:
    pickle.dump(ols_model, f)

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

# 1. Get predictions for both sets
y_pred_train = ols_model.predict(X_train_sm)
y_pred_val   = ols_model.predict(X_val_sm)

plt.figure(figsize=(7,7))

# 2. Plot Training Data (Blue)
plt.scatter(y_train, y_pred_train, 
            color='blue', 
            label=f'Training Data ({len(y_train)} samples)', 
            s=20, alpha=0.5)

# 3. Plot Testing Data (Orange)
plt.scatter(y_val, y_pred_val, 
            color='orange', 
            label=f'Testing Data ({len(y_val)} samples)', 
            s=30, alpha=0.7)

# 4. Determine min and max for the diagonal line
min_val = min(y_train.min(), y_val.min())
max_val = max(y_train.max(), y_val.max())
plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='Perfect Fit')

plt.xlabel("Actual Performance_Index")
plt.ylabel("Predicted Performance_Index")
plt.title("OLS: Predicted vs Actual (Train vs Test)")
plt.legend()
plt.grid(True)
plt.savefig("../documentation/OLS/pred_vs_actual_OLS_tennis_combined.png")
plt.show()
plt.close()
# ------------------------------
# Residuals vs Fitted plot
# ------------------------------

residuals = y_train - ols_model.fittedvalues


# 1. Calculate Required Diagnostic Values
influence = ols_model.get_influence()
std_residuals = influence.resid_studentized_internal
fitted_values = ols_model.fittedvalues
leverage = influence.hat_matrix_diag


plt.figure(figsize=(8, 5))
plt.scatter(fitted_values, ols_model.resid, alpha=0.6)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Fitted Values")
plt.ylabel("Residuals")
plt.title("Residual vs Fitted: Check for Non-linearity")
plt.grid(True)
plt.savefig("../documentation/OLS/diagnostic_resid_vs_fitted_tennis.png")
plt.close()


# Sqrt (Standardized Residual) vs Fitted values ---
plt.figure(figsize=(8, 5))
plt.scatter(fitted_values, np.sqrt(np.abs(std_residuals)), alpha=0.6, color='blue')

# Add a horizontal line at the mean to check for homoscedasticity
mean_sqrt_resid = np.mean(np.sqrt(np.abs(std_residuals)))
plt.axhline(y=mean_sqrt_resid, color='red', linestyle='--', label='Average Spread')

# OPTIONAL: Add a trend line to see if the spread "fans out"
z = np.polyfit(fitted_values, np.sqrt(np.abs(std_residuals)), 1)
p = np.poly1d(z)
plt.plot(fitted_values, p(fitted_values), "r-", alpha=0.8, label='Trend Line')

plt.xlabel("Fitted Values")
plt.ylabel("√|Standardized Residuals|")
plt.title("Scale-Location: Check for Homoscedasticity")
plt.legend()
plt.grid(True)
plt.savefig("../documentation/OLS/diagnostic_scale_location_tennis.png")
plt.show()

#  Standardized Residual vs Theoretical Quantile (Q-Q Plot) ---
plt.figure(figsize=(8, 5))
stats.probplot(std_residuals, dist="norm", plot=plt)
plt.title("Normal Q-Q: Check for Normality of Residuals")
plt.grid(True)
plt.savefig("../documentation/OLS/diagnostic_qq_plot_tennis.png")
plt.close()

# Residual vs Leverage ---
plt.figure(figsize=(8, 5))
plt.scatter(leverage, std_residuals, alpha=0.6)
plt.axhline(0, color='red', linestyle='--')
# This identifies points that might disproportionately influence the model
plt.xlabel("Leverage")
plt.ylabel("Standardized Residuals")
plt.title("Residual vs Leverage: Check for Influential Points")
plt.grid(True)
plt.savefig("../documentation/OLS/diagnostic_resid_vs_leverage_tennis.png")
plt.close()

print("All diagnostic plots generated successfully.")

