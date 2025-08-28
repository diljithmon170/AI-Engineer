from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

digits = load_digits()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.3)

# logistic regression
lr = LogisticRegression()
lr.fit(X_train, y_train)
print("Logistic Regression Test Score: ", lr.score(X_test, y_test))

# support vector machine
svm = SVC()
svm.fit(X_train, y_train)
print("SVM Test Score: ", svm.score(X_test, y_test))


# random forest
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
print("Random Forest Test Score: ", rf.score(X_test, y_test))


# KFold Cross Validation
# basic example

from sklearn.model_selection import KFold
kf = KFold(n_splits=3)
print(kf)

for train_index, test_index in kf.split([1, 2, 3, 4, 5, 6, 7, 8, 9]):
    print("Train Index: ", train_index, " Test Index: ", test_index)


# KFold with our digits data
def get_score(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    return model.score(X_test, y_test)

# use the cross validation score function
# cv = number of folds
# folds means how many parts to split the data into
# for each part, the model is trained on the other parts and tested on that part

from sklearn.model_selection import cross_val_score

print(cross_val_score(LogisticRegression(solver='liblinear', multi_class='ovr'), digits.data, digits.target, cv=3))
print(cross_val_score(SVC(gamma='auto'), digits.data, digits.target, cv=3))
print(cross_val_score(RandomForestClassifier(n_estimators=40), digits.data, digits.target, cv=3))