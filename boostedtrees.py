# -*- coding: UTF-8 -*-

from datetime import datetime
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
import sklearn.metrics

texts = []
labels = []
test_texts = []
test_labels = []

df = pd.read_csv('nonverbal.csv',
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
#classifier = GradientBoostingClassifier(learning_rate=0.001)
classifier = GradientBoostingClassifier()
Xs = vectorizer.fit_transform(texts)

print(datetime.now() - t1)
print(Xs.shape)

grid_param = {
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [1, 2, 3, 4]
}

#'F1': sklearn.metrics.make_scorer(sklearn.metrics.f1_score()
#scoring = {'AUC': sklearn.metrics.make_scorer(sklearn.metrics.roc_auc_score), 'Accuracy': sklearn.metrics.make_scorer(sklearn.metrics.accuracy_score)}
# sklearn.metrics.make_scorer(sklearn.metrics.cohen_kappa_score)
#'accuracy'
#'f1'
#sklearn.metrics.make_scorer(sklearn.metrics.roc_auc_score)

#score = cross_val_score(classifier, Xs, labels, scoring='f1', cv=10, n_jobs=4)

score = GridSearchCV(estimator = classifier, param_grid=grid_param, scoring=sklearn.metrics.make_scorer(sklearn.metrics.roc_auc_score), cv=10, n_jobs=4, refit=True)
model = score.fit(Xs, labels)

best_parameters = model.best_params_
print(best_parameters)

output = model.cv_results_
print(output)
