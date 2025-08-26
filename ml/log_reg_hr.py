import pandas as pd
from matplotlib import pyplot as plt

df = pd.read_csv("HR.csv")
# print(df.head())

# exployers left the company

left = df[df.left==1]
retained = df[df.left==0] # not left

#print(left.shape) # (3571, 10)
#print(retained.shape) # (11428, 10)

# average numbers for all columns
# print(df.groupby('left').mean())

pd.crosstab(df.salary, df.left).plot(kind='bar') # crosstab - cross tabulation
# plt.show()

pd.crosstab(df.Department, df.left).plot(kind='bar') # crosstab - cross tabulation
# plt.show()


# subset of dataframe with few columns for analysis
subdf = df[['satisfaction_level', 'average_montly_hours', 'promotion_last_5years', 'salary']]
# print(subdf.head())

#dummy variables
salary_dummies = pd.get_dummies(subdf.salary, prefix="salary").astype(int) # prefix - to avoid confusion
# print(salary_dummies.head())

# concat the two dataframes
merged = pd.concat([subdf, salary_dummies], axis='columns')
# print(merged.head())

# drop the salary column
final = merged.drop('salary', axis='columns')
# print(final.head())

# define x and y variables for the model
x= final
y= df.left
print(x)
print(y)

# create and train the model
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3) # 30% data for testing 


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()

model.fit(x_train, y_train)

# predict the price of a car with given features
print(model.predict(x_test))

# model score
print(model.score(x_test,y_test))