import pandas as pd

df = pd.read_csv("diabetes_data_upload.csv") 
print(df)

print("\nStatistical Summary:")
print(df.describe())

print("\nInterpretation:")
print("Mean:", df.mean(numeric_only=True))
print("Std Dev:", df.std(numeric_only=True))
print("Min:", df.min(numeric_only=True))
print("Max:", df.max(numeric_only=True))
