import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from datetime import date

data = pd.read_csv('./input/Combined_News_DJIA.csv')
# print(data.head())

train = data[data['Date'] < '2015-01-01']
test = data[data['Date'] > '2014-12-31']

X_train = train[train.columns[2:]]
corpus = X_train.values.flatten().astype(str)
X_train = X_train.values.astype(str)
X_train = np.array([' '.join(x) for x in X_train])

X_test = test[test.columns[2:]]
X_test = X_test.values.astype(str)
X_test = np.array([' '.join(x) for x in X_test])

y_train = train['Label'].values
y_test = test['Label'].values
print(X_train.head())
