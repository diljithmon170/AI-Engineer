import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("titanic.csv")
# print(df.head())

inputs = df[["Pclass", "Sex", "Age", "Fare"]]
targets = df["Survived"]

# print(inputs.head())
# print(targets.head())

dummies = pd.get_dummies(df["Sex"]).astype(int)
inputs = pd.concat([inputs, dummies], axis='columns')
print(inputs.head())
final = inputs.drop(["Sex","male"], axis='columns')

# print(final.head())

# check the null values in Age column
# print(final.isnull().sum())

# fill the null values with mean value of Age column
final["Age"].fillna(final["Age"].mean(), inplace=True)
# print(final.isnull().sum())

# round the age values and convert to integer
final["Age"] = final["Age"].round().astype(int)
# print(final.head())

# round the fare values and convert to integer
final["Fare"] = final["Fare"].round().astype(int)
print(final.head())

print(targets.head())

# create and train the model using Tree

from sklearn import tree
model = tree.DecisionTreeClassifier()

model.fit(final, targets)

# print(model.score(final, targets))

# predict the survival of a 3rd class

print(model.predict([[3, 22, 7, 0]]) ) # 0 means not survived

print(model.predict([[1, 35, 53, 1]]) ) # 1 means survived


# visualize the decision tree with labels
plt.figure(figsize=(15,10))
tree.plot_tree(model, filled=True, feature_names=final.columns, class_names=["Not Survived", "Survived"])
plt.show()