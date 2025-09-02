import pandas as pd
from sklearn.datasets import load_digits
from sklearn.neighbors import KNeighborsClassifier

digits = load_digits()
df = pd.DataFrame(digits.data, columns=digits.feature_names)

# train_test_split and manually tune parameters by trial and error
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.3, random_state=1)

# create the model of knn and fit the model
model = KNeighborsClassifier(n_neighbors=10)
model.fit(x_train,y_train)
print(model.score(x_test, y_test))

# confusion matrix
from sklearn.metrics import confusion_matrix
y_pred = model.predict(x_test)
print(confusion_matrix(y_test, y_pred))

# classification report - precision, recall, f1-score
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))