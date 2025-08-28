import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

iris = load_iris()

df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target

x = df.drop(['target'], axis='columns')
y = df.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

# cross val score function


print(cross_val_score(LogisticRegression(), x, y, cv=3))
print(cross_val_score(SVC(), x, y, cv=3))
print(cross_val_score(RandomForestClassifier(), x, y, cv=3))