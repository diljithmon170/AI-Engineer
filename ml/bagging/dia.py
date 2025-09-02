import pandas as pd

df = pd.read_csv('diabetes.csv')
# print(df.head())

# print(df.describe)
# print(df.Outcome.value_counts())

# tran test split

X = df.drop(columns=['Outcome'], axis='columns')
y = df.Outcome

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# print(X_scaled[:3])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, stratify=y, random_state=10)
# print(X_train.shape)

# train using stand alone model

from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

scores = cross_val_score(DecisionTreeClassifier(), X_train, y_train, cv=5)
# print(scores.mean())

# train using bagging model
from sklearn.ensemble import BaggingClassifier

bag_model = BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators=100,
    max_samples=0.8,
    oob_score=True,
    random_state=0
)
bag_model.fit(X_train, y_train)
print(bag_model.oob_score_)
print(bag_model.score(X_test, y_test))

# train using random forest model
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(n_estimators=100)
rf_model.fit(X_train, y_train)
print(rf_model.score(X_test, y_test))