import pandas as pd
df = pd.read_csv('data.csv')
columns = df.columns.tolist()
print(columns)
