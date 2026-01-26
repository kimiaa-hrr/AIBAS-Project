import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import pickle 



train_file = "../data/dataset/training_data.csv"
val_file   = "../data/dataset/test_data.csv"

# Load the datasets
df_train = pd.read_csv(train_file)
df_val   = pd.read_csv(val_file)

# Quick check
print("Training set shape:", df_train.shape)
print("Validation set shape:", df_val.shape)
