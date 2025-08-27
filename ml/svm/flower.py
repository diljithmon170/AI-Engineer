import pandas as pd
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

iris = load_iris()

df = pd.DataFrame(iris.data, columns=iris.feature_names)
# print(df.head())

df['target'] = iris.target
# print(df.head())

# set the flower names for the target values
df['flower_name'] = df.target.apply(lambda x: iris.target_names[x])
print(df.head())

# visualize the data
df0 = df[:50]
df1 = df[50:100]
df2 = df[100:]


# create a scatter plot for sepal length vs sepal width (setosa vs versicolor)
plt.xlabel('sepal length (cm)')
plt.ylabel('sepal width (cm)')
plt.scatter(df0['sepal length (cm)'], df0['sepal width (cm)'], color="red", marker='+')
plt.scatter(df1['sepal length (cm)'], df1['sepal width (cm)'], color="blue", marker='.')
# plt.show()

# create a scatter plot for petal length vs petal width (versicolor vs virginica)
plt.xlabel('petal length (cm)')
plt.ylabel('petal width (cm)')
plt.scatter(df1['petal length (cm)'], df1['petal width (cm)'], color="blue", marker='+')
plt.scatter(df2['petal length (cm)'], df2['petal width (cm)'], color="green", marker='.')
# plt.show()


# create and train the model using SVM

from sklearn.model_selection import train_test_split

x = df.drop(['target', 'flower_name'], axis='columns')
y = df.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

print(len(x_train), len(x_test))

# import SVC (Support Vector Classifier) from sklearn
from sklearn.svm import SVC
model = SVC()
model.fit(x_train, y_train)
print(model.score(x_test, y_test))  # accuracy of the model

# predict the flower type for a new sample
print(model.predict([[5.1, 3.5, 1.4, 0.2]]))  # 0 means setosa

print(model.predict([[4.8,3.0,1.5,0.3]]))


# tune the model by changing Regularization parameter C

model = SVC(C=10)  # C is the regularization parameter
model.fit(x_train, y_train)
print(model.score(x_test, y_test))  # accuracy of the model

# tune the gamma parameter
model = SVC(gamma=10)
model.fit(x_train, y_train)
print(model.score(x_test, y_test))  # accuracy of the model

# tune the kernel parameter
model = SVC(kernel='linear')
model.fit(x_train, y_train)
print(model.score(x_test, y_test))  # accuracy of the model
