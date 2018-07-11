import sys
import os
from numpy import *
import numpy as np
#from Nbayes_lib import *

def loadDataSet():
		classVec = [0,1,0,1,0,1]
		postingList =[['my','dog','has','flea','problem','help','please'],
		['maybe','not','take','him','to','dog','park','stupid'],
		['my','dalmation','is','so','cute','I','love','him','my'],
		['stop','posting','stupid','worthless','garbage'],
		['mr','licks','ate','my','steak','how','to','stop','him'],
		['quit','buying','worthless','dog','food','stupid']]

		return postingList,classVec

#编写一个贝叶斯算法类，并创建默认的构造方法
class NBayes(object) :
	def __init__(self) :
		self.vocabulary = []        #词典
		self.idf        = 0         #词典的IDF权值向量
		self.tf         = 0         #训练值的权值矩阵
		self.tdm        = 0         #P(x|yi)
		self.Pcates     = {}        #P(yi)是一个类别字典
		self.labels     = []        #对应每个文本的分类，是一个外部导入的列表
		self.doclength  = 0         #训练集文本数
		self.vocablen   = 0         #词典词长
		self.testset    = 0         #测试集

        
#导入和训练数据集，生成算法必需的参数和数据结构
	def train_set(self,trainset,classVec) :
		self.cate_prob(classVec)       #计算每个分类在数据集中的概率P(yi)
		self.doclength       = len(trainset)       #6
		tempset              = set()
		[tempset.add(word) for doc in trainset for word in doc]  #生成词典
		self.vocabulary      = list(tempset)                #转换成列表
		self.vocabulary.sort()
		self.vocablen        = len(self.vocabulary)            #  含有不重复词的个数为32
		self.calc_wordfreq(trainset)
		self.build_tdm() #按分类累计向量空间的每维值P(x|yi)

#cate_prob函数：计算在数据集中每个分类的概率P（yi）
	def cate_prob(self,classVec):
		self.labels          = classVec
		labeltemps           = set(self.labels)                   #获取全部分类 {0, 1}
		for labeltemp in labeltemps :
			#统计列表中重复的分类：  self.labels.count(labeltemp)   3个0,  3个1概率都是1/2
			self.Pcates[labeltemp] = float(self.labels.count(labeltemp))/float(len(self.labels))
			
	#calc_wordfreq函数：生成普通的词频向量
	def calc_wordfreq(self,trainset):
		self.idf             = np.zeros([1,self.vocablen])        #1*词典数 1x32的矩阵[[0,0,0,0,...,0]]   共32个0
		self.tf              = np.zeros([self.doclength,self.vocablen]) #训练集文件数*词典数 6x32的矩阵[[0,0,0,0,...,0]]
		for indx in range(self.doclength) :                     #遍历所有的文本  doclength=6
			for word in trainset[indx]:                                 #遍历文本中的每个词
				#找到文本的词在字典中的位置+1
				self.tf[indx,self.vocabulary.index(word)] += 1  #self.tf   训练值的权值矩阵
			for signleword in set(trainset[indx]):               #idf词数   
				self.idf[0,self.vocabulary.index(signleword)] += 1           #词典的IDF权值向量
   
#Build_tdm函数：按分类累计计算向量空间的每维值P（x|yi）
	def build_tdm(self):
		self.tdm = np.zeros([len(self.Pcates),self.vocablen])     #类别行*词典列 len({0: 0.5, 1: 0.5})=2，32
		sumlist  = np.zeros([len(self.Pcates),1])                #统计每个分类的总值
		for indx in range(self.doclength):                         #doclength=6
			#将同一类别的词向量空间值加总
			self.tdm[self.labels[indx]] += self.tf[indx]          #self.labels=[0, 1, 0, 1, 0, 1]
			#统计每个分类的总值--是一个标量
			sumlist[self.labels[indx]]    = np.sum(self.tdm[self.labels[indx]])   #sumlist=[[25],[19]]
#		print(sumlist)
		self.tdm = self.tdm/sumlist                              #生成P(x|yi)
#		print(self.tdm)

#map2vocab函数：将测试集映射到当前词典
	def map2vocab(self,testdata):
		self.testset = np.zeros([1,self.vocablen])                        #vocablen=32
		for word in testdata:
			self.testset[0,self.vocabulary.index(word)] += 1
#		print(self.testset)

#predict函数：预测分类结果，输出预测的分类类别
	def predict(self,testset):
#		print(np.shape(testset)[1])
		if np.shape(testset)[1] != self.vocablen: #如果测试集长度与词典不相等，则退出程序
			print ("输入错误")
			exit(0)
		predvalue = 0                             #初始化类别概率
		predclass = ""                            #初始化类别名称
		xyz = zip(self.tdm,self.Pcates)
#		print(list(xyz))
		for tdm_vect,keyclass in zip(self.tdm,self.Pcates):     #Pcates = {0: 0.5, 1: 0.5}
			#P(x|yi)*P(yi)
			#变量tdm，计算最大分类值
			temp = np.sum(testset*tdm_vect*self.Pcates[keyclass])
			print(testset)
			print(tdm_vect)
			print(temp)
			if temp > predvalue:
				predvalue = temp
				predclass = keyclass
		return predclass
'''	
#算法的改进
	def calc_tfidf(self,trainset):
		self.idf = no.zeros([1,self.vocablen])
		self.tf  = np.zeros([self.doclength,self.vocablen])
		for indx in xrange(self.doclength):
			for word in trainset[indx]:
				self.tf[indx,self.vocabulary.index(word)] += 1
			#消除不同句长导致的偏差
			self.tf[indx] = self.tf[indx]/float(len(trainset[indx]))
			for signleword in set(trainset[indx]):
				self.idf[0,self.vocabulary.index(signleword)] += 1
		self.idf = np.log(float(self.doclength)/self.idf)
		self.tf  = np.multiply(self.tf,self.idf)    #矩阵与向量的点乘 TF*IDF
'''
dataSet,listClasses = loadDataSet() #导入外部数据集
#dataSet:句子的词向量
#listClass是句子所属的类别[0,1,0,1,0,1]
nb = NBayes()                                     #实例化
nb.train_set(dataSet,listClasses)   #训练数据集
nb.map2vocab(dataSet[3])            #随机选择一个测试语句
print (nb.predict(nb.testset))
