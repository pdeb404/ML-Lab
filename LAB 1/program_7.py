# Load dataset, show structure, rows, columns & missing values

import pandas as pd

df = pd.read_csv("diabetes_data_upload.csv")  # replace with your dataset

print("Dataset Info:")
df.info()

print("\nNumber of Rows:", df.shape[0])
print("Number of Columns:", df.shape[1])

print("\nMissing Values in Each Column:")
print(df.isnull().sum())
