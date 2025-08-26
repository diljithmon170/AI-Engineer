import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
from word2number import w2n
## load data

df = pd.read_csv("hiring.csv")

# fill NaN with zero
df.experience = df.experience.fillna('zero')



# convert words to numbers
df.experience = df.experience.apply(w2n.word_to_num)


import math
median_test_score = math.floor(df['test_score(out of 10)'].mean())
median_test_score


df['test_score(out of 10)'] = df['test_score(out of 10)'].fillna(median_test_score)

print(df)

## create linear regression model

reg = linear_model.LinearRegression()
reg.fit(df[['experience', 'test_score(out of 10)', 'interview_score(out of 10)']], df['salary($)'])
print(reg.predict([[2, 9, 6]]))
print(reg.predict([[12, 10, 10]]))
# print(reg.coef_)
# print(reg.intercept_)
# y = m1*x1 + m2*x2 + m3*x3 + b
