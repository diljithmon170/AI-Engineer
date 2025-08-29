import pandas as pd

df = pd.read_csv('spam.csv')
# print(df.head())

df['spam'] = df['Category'].apply(lambda x:1 if x=='spam' else 0)
# print((df.head()))

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(df.Message,df.spam)

# convert the msg to vectors

from sklearn.feature_extraction.text import CountVectorizer
v = CountVectorizer()
x_train_count = v.fit_transform(x_train.values)
print(x_train_count.toarray()[:2])

# import the nb model 
from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(x_train_count,y_train)

# email passing and predict the output

emails = [
    'Hey mohan, can we get together to watch footbal game tomorrow?',
    'Upto 20% discount on parking, exclusive offer just for you. Dont miss this reward!'
]
emails_count = v.transform(emails)
print(model.predict(emails_count))

# score the accuracy of the model

x_train_count = v.transform(x_test)
print(model.score(x_train_count, y_test))

# using the pipeline to transform the msg into vectordb and use the model nb
from sklearn.pipeline import Pipeline
clf = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('nb', MultinomialNB())
])

clf.fit(x_train,y_train)

print(clf.score(x_test, y_test))

print(clf.predict(emails))