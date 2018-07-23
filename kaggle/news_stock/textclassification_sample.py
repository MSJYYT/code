from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
from datetime import date
from nltk.corpus import stopwords
#import stopwords
import re
from nltk.stem import WordNetLemmatizer

class test_classification(object):
    def __init__(self):
        #合并标签后的原始数据集
        self.data = []

        #原始数据集划分的训练、测试集
        self.train = []
        self.test = []

        #预处理后的训练、测试集无标签
        self.X_train = []
        self.X_test = []

        #训练、测试集标签
        self.y_train = []
        self.y_test = []
    def Text_preprocessing1(self):
        self.data = pd.read_csv('input/Combined_News_DJIA.csv')
        #print(data.head())
        self.data['combined_news'] = self.data.filter(regex=('Top.*')).apply(lambda x:''.join(str(x.values)),axis=1)
        self.train = self.data[self.data['Date'] < '2015-01-01']
        self.test = self.data[self.data['Date'] > '2014-12-31']

        feature_extraction = TfidfVectorizer()
        self.X_train = feature_extraction.fit_transform(self.train['combined_news'].values)
        self.X_test = feature_extraction.transform(self.test['combined_news'].values)

        self.y_train = self.train['Label'].values
        self.y_test = self.test['Label'].values



    def Text_preprocessing2(self):
        self.data = pd.read_csv('input/Combined_News_DJIA.csv')
        # print(data.head())
        self.data['combined_news'] = self.data.filter(regex=('Top.*')).apply(lambda x: ''.join(str(x.values)), axis=1)
        self.train = self.data[self.data['Date'] < '2015-01-01']
        self.test = self.data[self.data['Date'] > '2014-12-31']

        self.X_train = self.train['combined_news'].str.lower().str.replace('"','').str.replace("'",'').str.split()
        self.X_train = self.test['combined_news'].str.lower().str.replace('"','').str.replace("'",'').str.split()

        stop = stopwords.words('english')
        def hasNumbers(inputString):
            return bool(re.search(r'\d',inputString))
        wordnet_lemmatizer = WordNetLemmatizer()

        def check(word):
            if word in stop:
                return False
            elif hasNumbers(word):
                return False
            else:
                return True
        self.X_train = self.X_train.apply(lambda x:[wordnet_lemmatizer.lemmatize(item) for item in x if check(item)])
        self.X_test = self.X_test.apply(lambda x:[wordnet_lemmatizer.lemmatize(item) for item in x if check(item)])

        self.X_train = self.X_train.apply(lambda x:''.join(x))
        self.X_test = self.X_test.apply(lambda x:''.join(x))

        feature_extraction = TfidfVectorizer(lowercase=False)
        self.X_train = feature_extraction.fit_transform(self.X_train.values)
        self.X_test = feature_extraction.transform(self.X_test.values)

if __name__ == '__main__':
    classification = test_classification()
    # classification.Text_preprocessing1()
    # clf = SVC(probability=True,kernel='rbf')
    # clf.fit(classification.X_train,classification.y_train)
    # predictions = clf.predict_proba(classification.X_test)
    # print('ROC-AUC yields1=' + str(roc_auc_score(classification.y_test,predictions[:,1])))

    classification.Text_preprocessing2()
    clf = SVC(probability=True, kernel='rbf')
    clf.fit(classification.X_train, classification.y_train)
    predictions = clf.predict_proba(classification.X_test)
    print('ROC-AUC yields2=' + str(roc_auc_score(classification.y_test, predictions[:, 1])))


# import nltk
# nltk.download_gui()
