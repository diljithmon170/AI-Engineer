import pandas as pd
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

digits = load_digits()

# print(dir(digits))

# print(digits.feature_names[0])
# print(digits.data[0])
# print(digits.target[0])
# print(digits.target_names[0])

df = pd.DataFrame(digits.data)
# print(df.head())

df['target'] = digits.target
# print(df.head())


from sklearn.model_selection import train_test_split

x = df.drop('target', axis='columns')
# print(x.head())
y = df.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
# print(len(x_train), len(x_test))


# import SVC (Support Vector Classifier) from sklearn
from sklearn.svm import SVC
model = SVC()
model.fit(x_train, y_train)
print(model.score(x_test, y_test))  # accuracy of the model

print(model.predict(digits.data[0:5]))  # predict the first 5 samples