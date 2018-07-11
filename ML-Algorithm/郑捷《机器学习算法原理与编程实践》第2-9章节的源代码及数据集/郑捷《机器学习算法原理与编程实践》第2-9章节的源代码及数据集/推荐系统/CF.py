import sys
import os
from numpy import *
import numpy as np
import matplotlib.pyplot as plt
import operator

def loadDataSet():
	classVec = ['B', 'C', 'D','E', 'F']
	postingList =[[0.417,0.0,0.25,0.333],
	[0.3,0.4,0.0,0.3],
	[0.0,0.0,0.625,0.375],
	[0.278,0.222,0.222,0.278],
	[0.263,0.211,0.263,0.263]]
	return postingList,classVec

#夹角cos计算公式
def cosdist(vector1, vector2) :
	return dot(vector1, vector2) / (linalg.norm(vector1) * linalg.norm(vector2))
	
#KNN实现分类器
#测试集testdata; 训练集:trainSet; 类别标签：listClasses; k:k个邻居数
def classify(testdata, trainSet, labels) :
	dataSetSize = mat(trainSet).shape[0]                                                      #返回样本集的行数  5
	distances = array(zeros(dataSetSize))                                           #[ 0.  0.  0.  0.  0.]
#	print(trainSet[0])
	for indx in range(dataSetSize) :                                                    #dataSetSize=5
		distances[indx] = cosdist(testdata, trainSet[indx])
	bestDistIndicies = argsort(-distances)[0]
	return labels[bestDistIndicies]
	
#算法测试
dataSet, labels = loadDataSet()
testSet = [0.334,0.333,0.0,0.333]
print(classify(testSet, dataSet, labels))
		




