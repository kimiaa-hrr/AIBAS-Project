import pandas as pd
from sklearn.preprocessing import MinMaxScaler 

df = pd.read_csv("../data/dataset/scrapedDataset.csv")


# ============== Handling Missing Data ==============


# filling the NA for athelete names with unknown
# df["Athlete_Name"]=df["Athlete_Name"].fillna("Unknown")



# df = df.dropna(subset=['Athlete_ID'])
df = df.dropna(subset=['Sport_Type'])
# df = df.dropna(subset=['Event'])

# Only replace actual missing values, not the string "None"
df["Injury_History"] = df["Injury_History"].fillna("None")

# Clean up whitespace/capitalization
df["Injury_History"] = df["Injury_History"].str.strip().str.capitalize()



# Fill numeric columns with median
numeric_cols = df.select_dtypes(include='number').columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

# Fill categorical/text columns with mode
cat_cols = df.select_dtypes(include='string').columns  # only string columns
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])





# the Athlete_id seems to have duplicates so I added a new column for Ids so it is unique
df['Athlete_ID_New'] = ['A{:03d}'.format(i+1) for i in range(len(df))]
# Remove the column from its current position and insert at index 0
col = df.pop('Athlete_ID_New')
df.insert(0, 'Athlete_ID_New', col)




# Drop columns
df=df.drop(columns=["Athlete_ID","Event","Athlete_Name","Competition_Date","Athlete_ID_New"])


# ============== Drop outliers with mathematical methods ==============

# Apply IQR filter to each numeric column
for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    
    # Keep rows within the IQR bounds
    df = df[(df[col] >= Q1 - 1.5*IQR) & (df[col] <= Q3 + 1.5*IQR)]

# Check how many rows are left
print(f"Number of rows after removing outliers: {len(df)}")




# ============== Data Normalizing ==============

scaler = MinMaxScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])




# ============== Data Encoding ==============

# Check unique values
print(df["Injury_History"].unique())

#clean whitespace/capitalization
df["Injury_History"] = df["Injury_History"].str.strip().str.capitalize()

# Convert to Categorical with ordered categories
df["Injury_History"] = pd.Categorical(
    df["Injury_History"],
    categories=["None", "Minor", "Major"],
    ordered=True
)




# Check unique values
print(df["Sport_Type"].unique())

#clean whitespace/capitalization
df["Sport_Type"] = df["Sport_Type"].str.strip().str.capitalize()

# Convert to Categorical with ordered categories
df["Sport_Type"] = pd.Categorical(
    df["Sport_Type"],
    categories=['Running', 'Swimming', 'Cycling', 'Soccer', 'Basketball', 'Tennis'],
    ordered=True
)



# Check unique values
print(df["Training_Intensity"].unique())

#clean whitespace/capitalization
df["Training_Intensity"] = df["Training_Intensity"].str.strip().str.capitalize()

# Convert to Categorical with ordered categories
df["Training_Intensity"] = pd.Categorical(
    df["Training_Intensity"],
    categories=['Low', 'High', 'Medium'],
    ordered=True
)




# Check unique values
print(df["Altitude_Training"].unique())

#clean whitespace/capitalization
df["Altitude_Training"] = df["Altitude_Training"].str.strip().str.capitalize()

# Convert to Categorical with ordered categories
df["Altitude_Training"] = pd.Categorical(
    df["Altitude_Training"],
    categories=['Low', 'High', 'Medium'],
    ordered=True
)


df_encoded = pd.get_dummies(df, columns=["Injury_History","Sport_Type","Training_Intensity","Altitude_Training"], drop_first=True)
df = df_encoded 




# Save cleaned dataset
df.to_csv("../data/dataset/joint_data_collection.csv", index=False)
print("Saved cleaned_dataset")
