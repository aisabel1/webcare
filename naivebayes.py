# -*- coding: UTF-8 -*-

from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
import sklearn.metrics

texts = []
labels = []
test_texts = []
test_labels = []

df = pd.read_csv('sigstaff.csv',
    header=None,
    # names=['label', 'text'],
    names=['label', 'text'],
    nrows=50000,
    encoding='UTF-8')

for i in range (len(df.label)):
    texts.append(df['text'][i])
    labels.append(df['label'][i])

t1 = datetime.now()
vectorizer = CountVectorizer()
classifier = MultinomialNB()
Xs = vectorizer.fit_transform(texts)

print(datetime.now() - t1)
print(Xs.shape)

#'accuracy'
#'f1'
#sklearn.metrics.make_scorer(sklearn.metrics.roc_auc_score)

score = cross_val_score(classifier, Xs, labels, scoring=sklearn.metrics.make_scorer(sklearn.metrics.roc_auc_score), cv=10, n_jobs=4)

predict = cross_val_predict(classifier, Xs, labels, cv=10, n_jobs=4)

print(score)
print(np.average(score))