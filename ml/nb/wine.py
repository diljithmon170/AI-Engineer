import pandas as pd
from sklearn.datasets import load_wine

wine = load_wine()
print(dir(wine))

df = pd.DataFrame(wine.data, columns=wine.feature_names)
df['target'] = wine.target

print(df.head())

x = df.drop('target', axis='columns')
y = df.target

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
model = GaussianNB()
model2 = MultinomialNB()

model.fit(x_train, y_train)
model2.fit(x_train, y_train)

print(model.score(x_test, y_test))
print(model2.score(x_test, y_test))

from sklearn.model_selection import cross_val_score
print(cross_val_score(GaussianNB(), x_test, y_test, cv=5))
print(cross_val_score(MultinomialNB(), x_test, y_test, cv=5))