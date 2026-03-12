# Create a Pandas DataFrame using dictionary & print data types

import pandas as pd

data = {
    "Student_Name": ["Aman", "Riya", "Kunal", "Neha"],
    "Marks": [85, 92, 78, 88]
}

df = pd.DataFrame(data)

print("DataFrame:")
print(df)

print("\nData Types:")
print(df.dtypes)
