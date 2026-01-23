import pandas as pd
from sklearn.preprocessing import MinMaxScaler 

df = pd.read_csv("../data/dataset/scrapedDataset.csv")


# ============== Handling Missing Data ==============


# filling the NA for athelete names with unknown
df["Athlete_Name"]=df["Athlete_Name"].fillna("Unknown")



df = df.dropna(subset=['Athlete_ID'])
df = df.dropna(subset=['Sport_Type'])
df = df.dropna(subset=['Event'])




# Fill numeric columns with median
numeric_cols = df.select_dtypes(include='number').columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

# Fill categorical/text columns with mode
cat_cols = df.select_dtypes(include='string').columns  # only string columns
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])


# the Athlete_id seems to have duplicates so I added a new column for Ids so it is unique
df['Athlete_ID_New'] = ['A{:03d}'.format(i+1) for i in range(len(df))]
print(df)





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





# Save cleaned dataset
df.to_csv("../data/dataset/joint_data_collection.csv", index=False)
print("Saved cleaned_dataset.csv")
