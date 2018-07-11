import sys
import os
from numpy import *
import numpy as np
import matplotlib.pyplot as plt
import operator

def loadDataSet():
		classVec = [0,1,0,1,0,1]
		postingList =[['my','dog','has','flea','problem','help','please'],
		['maybe','not','take','him','to','dog','park','stupid'],
		['my','dalmation','is','so','cute','I','love','him','my'],
		['stop','posting','stupid','worthless','garbage'],
		['mr','licks','ate','my','steak','how','to','stop','him'],
		['quit','buying','worthless','dog','food','stupid']]
		return postingList,classVec

#calc_wordfreq函数：生成普通的词频向量
def calc_tf(trainset, classVec):
	tempset              = set()
	[tempset.add(word) for doc in trainset for word in doc]  #生成词典
	vocabulary      = list(tempset)                                                        #转换成列表
	vocablen = len(vocabulary)
	doclength = len(trainset)
	tf  = np.zeros([doclength,vocablen])                                           #训练集文件数*词典数 6x32的矩阵[[0,0,0,0,...,0]]
	for indx in range(doclength) :                                                    #遍历所有的文本  doclength=6
		for word in trainset[indx]:                                                        #遍历文本中的每个词
			tf[indx,vocabulary.index(word)] += 1                                 #tf   训练值的权值矩阵
	return tf

#夹角cos计算公式
def cosdist(vector1, vector2) :
	return dot(vector1, vector2) / (linalg.norm(vector1) * linalg.norm(vector2))
	
#KNN实现分类器
#测试集testdata; 训练集:trainSet; 类别标签：listClasses; k:k个邻居数
def classify(testdata, trainSet, labels, k) :
	dataSetSize = trainSet.shape[0]                                                      #返回样本集的行数  6
	distances = array(zeros(dataSetSize))                                           #[ 0.  0.  0.  0.  0.  0.]
	for indx in range(dataSetSize) :                                                    #dataSetSize=6
		distances[indx] = cosdist(testdata, trainSet[indx])
	sortedDistIndicies = argsort(-distances)
	classCount = {}
	for i in range(k) :                                                                                  #获取角度最小的前K项作为参考项
		voteIlabel = labels[sortedDistIndicies[i]]                                   #按排列顺序返回样本集对应的类别标签
		classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1  #为字典classCount赋值，相同的key，其value加1
	sortedClassCount = sorted(classCount.items(), key = operator.itemgetter(1),reverse = True)
	return sortedClassCount[0][0]
	
#算法测试
k = 3
dataSet, labels = loadDataSet()
tf = calc_tf(dataSet, labels)
print(classify(tf[2], tf, labels, k))
		
