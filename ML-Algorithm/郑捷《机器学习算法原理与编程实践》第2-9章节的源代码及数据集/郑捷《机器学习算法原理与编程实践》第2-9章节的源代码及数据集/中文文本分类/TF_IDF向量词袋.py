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
	
def writebunchobj(path, bunchobj) :
	file_obj = open(path, 'wb')
	pickle.dump(bunchobj, file_obj)
	file_obj.close()

def readfile(path) :
	fp = open(path, 'rb')
	content = fp.read()
	fp.close()
	return content

#读取停用词表
stopword_path = "train_word_bag/hlt_stop_words.txt"
stpwrdlst = readfile(stopword_path).splitlines()

#从训练集生成TD_IDF向量词袋
path = "train_word_bag/train_set.dat"                                           #词向量空间保存路径
bunch = readbunchobj(path)

#构建TF_IDF词向量空间对象
tfidspace = Bunch(target_name = bunch.target_name, label = bunch.label, filenames = bunch.filenames, tdm=[],vocabulary={})

#使用TfidfVectorizer初始化向量空间模型
vectorizer = TfidfVectorizer(stop_words=stpwrdlst, sublinear_tf = True, max_df = 0.5)
transformer = TfidfTransformer()

#文本转为词频矩阵，单独保存字典文件
tfidspace.tdm = vectorizer.fit_transform(bunch.contents)
tfidspace.vocabulary = vectorizer.vocabulary_

#持久化TD_IDF向量词袋
space_path = "train_word_bag/tfdifspace.dat"
writebunchobj(space_path, tfidspace)

print("结束")
