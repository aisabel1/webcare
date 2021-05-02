# -*- coding: UTF-8 -*-

from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score
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
vectorizer = TfidfVectorizer(analyzer= 'char', encoding='UTF-8', strip_accents='unicode', ngram_range=(1, 7), min_df=3)
classifier = RandomForestClassifier()
Xs = vectorizer.fit_transform(texts)

print(datetime.now() - t1)
print(Xs.shape)

#'accuracy'
#'f1'
#sklearn.metrics.make_scorer(sklearn.metrics.roc_auc_score)

score = cross_val_score(classifier, Xs, labels, scoring=sklearn.metrics.make_scorer(sklearn.metrics.roc_auc_score), cv=10, n_jobs=4)

print(score)
print(np.average(score))

