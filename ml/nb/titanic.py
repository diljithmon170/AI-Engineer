import pandas as pd

df = pd.read_csv('titanic.csv')
# print(df.head())

df.drop(['PassengerId', 'Name', 'Embarked', 'SibSp', 'Parch', 'Ticket', 'Cabin'], axis='columns', inplace=True)
# print(df.head())

input = df.drop(['Survived'], axis='columns')

dummies = pd.get_dummies(df.Sex).astype(int)
# print(dummies.head())

input = pd.concat([input,dummies], axis='columns')
# print(input.head())


input.drop(['Sex','male'], axis='columns', inplace=True)
print(input.head())

# print(input.columns[input.isna().any()])
input.Age = input.Age.fillna(input.Age.mean())

output = df.Survived
print(output.head())

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(input,output, test_size=0.3)

from sklearn.naive_bayes import GaussianNB
model = GaussianNB()

model.fit(x_train,y_train)

print(model.score(x_test,y_test))

# Calculate the score using cross validation

from sklearn.model_selection import cross_val_score
cross_val_score(GaussianNB(),x_train, y_train, cv=5)