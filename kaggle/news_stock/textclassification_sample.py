from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
from datetime import date
from nltk.corpus import stopwords
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
        #把headlines合并，所有的news合并一起
        self.data["combined_news"] = self.data.filter(regex=("Top.*")).apply(lambda x:''.join(str(x.values)),axis=1)
        #分割训练集与测试集
        self.train = self.data[self.data['Date'] < '2015-01-01']
        self.test = self.data[self.data['Date'] > '2014-12-31']

        feature_extraction = TfidfVectorizer()
        # fit原义指的是安装、使适合的意思，其实有点train的含义但是和train不同的是，它并不是一个训练的过程，
        # 而是一个适配的过程，过程都是定死的，最后只是得到了一个统一的转换的规则模型。
        # transform：是将数据进行转换，比如数据的归一化和标准化，将测试数据按照训练数据同样的模型进行转换，得到特征向量。
        # fit_transform：可以看做是fit和transform的结合，如果训练阶段使用fit_transform，则在测试阶段只需要对测试样本进行transform就行了。
        self.X_train = feature_extraction.fit_transform(self.train["combined_news"].values)
        self.X_test = feature_extraction.transform(self.test["combined_news"].values)

        self.y_train = self.train["Label"].values
        self.y_test = self.test["Label"].values



    def Text_preprocessing2(self):
        self.data = pd.read_csv('input/Combined_News_DJIA.csv')
        # print(data.head())
        self.data["combined_news"] = self.data.filter(regex=("Top.*")).apply(lambda x: ''.join(str(x.values)), axis=1)
        self.train = self.data[self.data['Date'] < '2015-01-01']
        self.test = self.data[self.data['Date'] > '2014-12-31']

        self.X_train = self.train["combined_news"].str.lower().str.replace('"', '').str.replace("'", '').str.split()
        self.X_test = self.test["combined_news"].str.lower().str.replace('"', '').str.replace("'", '').str.split()

        #删除停止词
        stop = stopwords.words('english')
        #删除数字
        def hasNumbers(inputString):
            return bool(re.search(r'\d',inputString))
        #lemma
        wordnet_lemmatizer = WordNetLemmatizer()

        def check(word):
            if word in stop:
                return False
            elif hasNumbers(word):
                return False
            else:
                return True
        #Lemmatization 把一个任何形式的语言词汇还原为一般形式，例如cars还原为car，feet还原为foot等等
        self.X_train = self.X_train.apply(lambda x:[wordnet_lemmatizer.lemmatize(item) for item in x if check(item)])
        self.X_test = self.X_test.apply(lambda x:[wordnet_lemmatizer.lemmatize(item) for item in x if check(item)])
        #变回string
        self.X_train = self.X_train.apply(lambda x:''.join(x))
        self.X_test = self.X_test.apply(lambda x:''.join(x))

        feature_extraction = TfidfVectorizer(lowercase=False)
        self.X_train = feature_extraction.fit_transform(self.X_train.values)
        self.X_test = feature_extraction.transform(self.X_test.values)

        self.y_train = self.train["Label"].values
        self.y_test = self.test["Label"].values

if __name__ == '__main__':
    classification = test_classification()
    classification.Text_preprocessing1()
    #训练模型
    clf = SVC(probability=True,kernel='rbf')
    clf.fit(classification.X_train,classification.y_train)
    #预测
    predictions = clf.predict_proba(classification.X_test)
    #验证准确度，用AUC做binary classification 的metrics
    print('ROC-AUC yields1=' + str(roc_auc_score(classification.y_test,predictions[:,1])))

    classification.Text_preprocessing2()
    clf = SVC(probability=True, kernel='rbf')
    clf.fit(classification.X_train, classification.y_train)
    predictions = clf.predict_proba(classification.X_test)
    print('ROC-AUC yields2=' + str(roc_auc_score(classification.y_test, predictions[:, 1])))

