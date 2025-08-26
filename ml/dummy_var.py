import pandas as pd
df = pd.read_csv("carprices.csv")
# print(df)

# create dummy variables as 0 and 1
dummies = pd.get_dummies(df.CarModel).astype(int)
print(dummies)