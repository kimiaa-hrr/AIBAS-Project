import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# ================= Load data =================
df = pd.read_csv("../data/dataset/scrapedDataset.csv")

df = df[df['Sport_Type'] == 'Tennis'].copy() 


# ================= Drop Values =================

# Drop rows with missing target or key grouping variable
df = df.dropna(subset=["Sport_Type", "Performance_Metric"])

# Drop irrelevant / ID columns
df = df.drop(columns=[
    "Athlete_ID",
    "Event",
    "Athlete_Name",
    "Competition_Date"
], errors="ignore")



# ================= Handle missing data =================


df = df.dropna()

# # Handle missing values in categorical columns
# cat_cols = df.select_dtypes(include="string").columns  # or "string" if using pandas >= 2.0
# for col in cat_cols:
#     df[col] = df[col].fillna(df[col].mode()[0])  # fill with most frequent value

# # Handle missing values in numeric columns
# num_cols = df.select_dtypes(include="number").columns
# for col in num_cols:
#     df[col] = df[col].fillna(df[col].median())  # fill with median to avoid outliers

# ================= Outlier removal (IQR, inputs only) =================

# Recompute numeric columns after row drops
numeric_cols = df.select_dtypes(include="number").columns

for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    df = df[
        (df[col] >= Q1 - 1.5 * IQR) &
        (df[col] <= Q3 + 1.5 * IQR)
    ]

print(f"Rows after outlier removal: {len(df)}")

# ================= Normalizing =================


# Inputs only (exclude target)
input_features = [col for col in numeric_cols if col != "Performance_Metric"]

scaler = MinMaxScaler()
df[input_features] = scaler.fit_transform(df[input_features])


#Scale Output (Target)
scaler_y = MinMaxScaler()
df[["Performance_Metric"]] = scaler_y.fit_transform(df[["Performance_Metric"]])

# ================= Encoding =================

# Injury history (ordered)
df["Injury_History"] = pd.Categorical(
    df["Injury_History"],
    categories=["None", "Minor", "Major"],
    ordered=True
)

# Training intensity
df["Training_Intensity"] = (
    df["Training_Intensity"]
    .str.strip()
    .str.capitalize()
)
df["Training_Intensity"] = pd.Categorical(
    df["Training_Intensity"],
    categories=["Low", "Medium", "High"],
    ordered=True
)

# Altitude training
df["Altitude_Training"] = (
    df["Altitude_Training"]
    .str.strip()
    .str.capitalize()
)
df["Altitude_Training"] = pd.Categorical(
    df["Altitude_Training"],
    categories=["Low", "Medium", "High"],
    ordered=True
)

# One-hot encode categoricals
df = pd.get_dummies(
    df,
    columns=[
        "Injury_History",
        "Sport_Type",
        "Training_Intensity",
        "Altitude_Training"
    ],
    drop_first=True,
    dtype=int
)

# ================= Save cleaned dataset =================
df.to_csv("../data/dataset/joint_data_collection_tennis.csv", index=False)
print("Saved cleaned_dataset")
