import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt


## load data

df = pd.read_csv("canada_per_capita_income.csv")
print(df.head())

plt.xlabel('year')
plt.ylabel('per capita income (US$)')
plt.scatter(df.year, df['per capita income (US$)'], color='red', marker='+')
# plt.show()

## create linear regression model

reg = linear_model.LinearRegression()
reg.fit(df[['year']], df['per capita income (US$)'])
print(reg.predict([[2020]]))

# print(reg.coef_)
# print(reg.intercept_)
# y = mx + b
# y = 828.46507522 * x - 1632210.75785546
