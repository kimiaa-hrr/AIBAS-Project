from sklearn.model_selection import train_test_split
import pandas as pd


df = pd.read_csv("../data/dataset/joint_data_collection_running.csv")



training_data, test_data = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    shuffle=True
)


training_data.to_csv("../data/dataset/training_data_running.csv", index=False)
test_data.to_csv("../data/dataset/test_data_running.csv", index=False)


print("saved training_data")
print("saved test_data") 