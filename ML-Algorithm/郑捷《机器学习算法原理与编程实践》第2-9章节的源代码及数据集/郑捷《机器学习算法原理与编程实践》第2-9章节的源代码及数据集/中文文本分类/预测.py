from sklearn.naive_bayes import MultinomialNB   #导入多项式贝叶斯算法包
import os
import sys
import pickle                                                                                                           #引入持久化类
from sklearn.datasets.base import Bunch                                                 #引入Bunch类
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer           #TF-TDF向量转换类
from sklearn.feature_extraction.text import TfidfVectorizer               #TF-IDF向量生成类

#读取和写入Bunch对象的函数
def readbunchobj(path) :
	file_obj = open(path, 'rb')
	bunch = pickle.load(file_obj)
	file_obj.close()
	return bunch

#导入训练集向量空间
trainpath = "train_word_bag/tfdifspace.dat"
train_set = readbunchobj(trainpath)

#导入测试集向量空间
testpath = "test_word_bag/testspace.dat"
test_set = readbunchobj(testpath)

#应用朴素贝叶斯算法
clf = MultinomialNB(alpha = 0.01).fit(train_set.tdm, train_set.label)

#预测分类结果
predicted = clf.predict(test_set.tdm)
total = len(predicted) 
rate = 0
for flabel,file_name, expct_cate in zip(test_set.label, test_set.filenames, predicted) :
	if flabel != expct_cate :
		rate += 1
		print(file_name, ": 实际类别：", flabel, "-->预测类别：", expct_cate)
#精度
print ("error rate:", float(rate) * 100/float(total), "%")

