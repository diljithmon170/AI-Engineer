import pandas as pd
import numpy as np

df = pd.read_csv("data.csv")

print(df.describe())
print(df.tail())

df1 = pd.DataFrame(np.arange(0,20).reshape(5,4),index=['row1', 'row2', 'row3', 'row4', 'row5'], columns=['col1', 'col2', 'col3', 'col4'])
print(df1.loc['row1'])  # Slicing rows and columns

print(df1.iloc[:,1:].values)

print(df1['col1'].value_counts())  # Count unique values in 'col1'

print(df1.isnull().sum())  # Check for null values in the DataFrame