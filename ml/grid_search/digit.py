from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
import pandas as pd
from sklearn.datasets import load_digits

digits = load_digits()
df = pd.DataFrame(digits.data, columns=digits.feature_names)

#Approach 1: Use train_test_split and manually tune parameters by trial and error
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.3)

model = svm.SVC(kernel='rbf',C=30,gamma='auto')
model.fit(x_train,y_train)
print(model.score(x_test, y_test))

#Approach 2: Use K Fold Cross validation

cross_val_score(svm.SVC(kernel='linear',C=10,gamma='auto'),digits.data,digits.target, cv=5)

cross_val_score(svm.SVC(kernel='rbf',C=10,gamma='auto'),digits.data,digits.target, cv=5)

cross_val_score(svm.SVC(kernel='rbf',C=20,gamma='auto'),digits.data,digits.target, cv=5)

#Approach 3: Use GridSearchCV
from sklearn.model_selection import GridSearchCV
clf = GridSearchCV(svm.SVC(gamma='auto'), {
    'C': [1,10,20],
    'kernel': ['rbf','linear']
}, cv=5, return_train_score=False)
clf.fit(digits.data, digits.target)
clf.cv_results_

df = pd.DataFrame(clf.cv_results_)
print(df[['param_C','param_kernel','mean_test_score']])

clf.best_params_

clf.best_score_

#Use RandomizedSearchCV to reduce number of iterations and with random combination of parameters. This is useful when you have too many parameters to try and your training time is longer. It helps reduce the cost of computation
from sklearn.model_selection import RandomizedSearchCV
rs = RandomizedSearchCV(svm.SVC(gamma='auto'), {
        'C': [1,10,20],
        'kernel': ['rbf','linear']
    }, 
    cv=5, 
    return_train_score=False, 
    n_iter=2
)
rs.fit(digits.data, digits.target)
df_rnd = pd.DataFrame(rs.cv_results_)[['param_C','param_kernel','mean_test_score']]
print(df_rnd)

# Different models with diff params

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

model_params = {
    'svm': {
        'model': svm.SVC(gamma='auto'),
        'params' : {
            'C': [1,10,20],
            'kernel': ['rbf','linear']
        }  
    },
    'random_forest': {
        'model': RandomForestClassifier(),
        'params' : {
            'n_estimators': [1,5,10]
        }
    },
    'logistic_regression' : {
        'model': LogisticRegression(solver='liblinear',multi_class='auto'),
        'params': {
            'C': [1,5,10]
        }
    }
}

scores = []
for model_name, mp in model_params.items():
    clf =  GridSearchCV(mp['model'], mp['params'], cv=5, return_train_score=False)
    clf.fit(digits.data, digits.target)
    scores.append({
        'model': model_name,
        'best_score': clf.best_score_,
        'best_params': clf.best_params_
    })
    
df = pd.DataFrame(scores,columns=['model','best_score','best_params'])
print(df)

# best model with param  svm    0.947697  {'C': 1, 'kernel': 'linear'}