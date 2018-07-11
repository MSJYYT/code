#评估分类结果
#coding:utf-8
import sys
import os
from numpy import *
import numpy as np
from Nbayes_lib import *

dataSet,listClasses = loadDataSet() #导入外部数据集
#dataSet:句子的词向量
#listClass是句子所属的类别[0,1,0,1,0,1]
nb = NBayes()                                     #实例化
nb.train_set(dataSet,listClasses)   #训练数据集
nb.map2vocab(dataSet[0])            #随机选择一个测试语句
print (nb.predict(nb.testset))
