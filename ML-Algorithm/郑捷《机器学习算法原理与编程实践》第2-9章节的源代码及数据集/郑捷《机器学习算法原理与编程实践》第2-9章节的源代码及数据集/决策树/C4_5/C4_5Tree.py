from numpy import *
import math
import copy
import pickle

class C4_5DTree(object) :
	def __init__(self) :                                                                                      #构造方法
		self.tree = {}                                                                                           #生成的树
		self.dataSet = []                                                                                    #数据集
		self.labels = []                                                                                        #标签集
		
	def loadDataSet(self, path, labels) :
		recordlist = []
		fp = open(path, 'rb')
		content = fp.read()
		fp.close()
		rowlist = content.splitlines()                                                              #按行转换成一维表
		recordlist = [row.split("\t".encode()) for row in rowlist if row.strip()]
		self.dataSet = recordlist
		self.labels = labels
		
		
	def train(self) :
		labels = copy.deepcopy(self.labels)
		self.tree = self.buildTree(self.dataSet, labels)
		
	def buildTree(self, dataSet, labels) :
		cateList = [data[-1] for data in dataSet]
		if cateList.count(cateList[0]) == len(cateList) :
			return cateList[0]
		if len(dataSet[0]) == 1 :
			return self.maxCate(cateList)
		bestFeat, featValueList = self.getBestFeat(dataSet)
		bestFeatLabel = labels[bestFeat]
		tree = {bestFeatLabel : {}}
		del(labels[bestFeat])
		for value in featValueList :
			subLabels = labels[:]
			splitDataSet = self.splitDataSet(dataSet, bestFeat, value)
			subTree = self.buildTree(splitDataSet,subLabels)
			tree[bestFeatLabel][value] = subTree
		return tree
		
				
	#计算出现次数最多的类别标签
	def maxCate(self, catelist) :
		items = dict([(cateList.count(i), i) for i in catelist])
		return items[max(items.keys())]
	
			#计算信息熵
	def computeEntropy(self, dataSet) :                                                      #计算香农熵
		datalen = float(len(dataSet))
		cateList = [data[-1] for data in dataSet]                                             #从数据集中得到类别标签
		#得到类别为key、出现次数value的字典
		items = dict([(i, cateList.count(i)) for i in cateList])                        #用法不错
#		print(items)
		infoEntropy = 0.0
		for key in items :
			prob = float(items[key]) /datalen
			infoEntropy -= prob * math.log(prob, 2)
		return infoEntropy

		#划分数据集；分隔数据集；删除特征轴所在的数据列， 返回剩余的数据集
		#dataSet:数据集；axis：特征轴；value：特征轴取值
	def splitDataSet(self, dataSet, axis, value) :
		rtnList = []
		for featVec in dataSet :
			if featVec[axis] == value :
				rFeatVec = featVec[:axis]                                                                 #list操作：提取0 - （axis - 1）的元素
				rFeatVec.extend(featVec[axis + 1 :])
				rtnList.append(rFeatVec)
		return rtnList

	#计算划分信息
	def computeSplitInfo(self, featureVList) :
		numEntries = len(featureVList)                                                          #1024,1024,540...
#		print(numEntries)
		featureValueSetList = list(set(featureVList))                                  #['0', '1', '2']                                                          
		valueCounts = [featureVList.count(featVec) for featVec in featureValueSetList]      #[384, 256, 384]
		pList = [float(item)/numEntries for item in valueCounts]         #[0.375, 0.25, 0.375]
		lList = [item*math.log(item,2) for item in pList]
		splitInfo = -sum(lList)
		return splitInfo, featureValueSetList
				
	#计算最优特征
	def getBestFeat(self, dataSet) :
		Num_Feats = len(dataSet[0][:-1])                                                          #4
#		print(Num_Feats)
		totality = len(dataSet)                                                                               #1024
#		print(totality)
		BaseEntropy = self.computeEntropy(dataSet)
		ConditionEntropy = []
		splitInfo = []
		allFeatVList = []
		for f in range(Num_Feats) :
			featList = [example[f] for example in dataSet]
			[splitI, featureValueList] = self.computeSplitInfo(featList)
			allFeatVList.append(featureValueList)                                             #['0','1','2'],['0','1','2'],['0','1']
			splitInfo.append(splitI)
#			print(splitInfo)
			resultGain = 0.0
			for value in featureValueList :
				subSet = self.splitDataSet(dataSet, f, value)
				appearNum = float(len(subSet))
				subEntropy = self.computeEntropy(subSet)
				resultGain += (appearNum/totality) * subEntropy
			ConditionEntropy.append(resultGain)
		infoGainArray = BaseEntropy * ones(Num_Feats) - array(ConditionEntropy)
		infoGainRatio = infoGainArray/(array(splitInfo)+(-inf))
		bestFeatureIndex = argsort(-infoGainRatio)[0]
		return bestFeatureIndex, allFeatVList[bestFeatureIndex]
		
		
	def predict(self, inputTree, featLabels, testVec) :                               #分类器
		root = list(inputTree.keys())[0]
#		print(root)                                                                                                    #树根节点
#		print(inputTree.keys())
		secondDict = inputTree[root]                                                                #value-子树结构或分类标签
#		print(secondDict)
		featIndex = featLabels.index(root)                                                     #根节点在分类标签集中的位置
#		print(featIndex)                                                                                         #0,2
		key = testVec[featIndex]
		valueOfFeat = secondDict[key.encode()]                                           #易错点valueOfFeat = secondDict[key]    
#		print(valueOfFeat)                                                                                    #{'student': {b'0': b'no', b'1': b'yes'}}
		if isinstance(valueOfFeat, dict) :
			classLabel = self.predict(valueOfFeat, featLabels, testVec)      #递归分类
		else :
			classLabel = valueOfFeat
		return classLabel
			
	def storeTree(self, inputTree, filename) :
		fw = open(filename, 'wb')
		pickle.dump(inputTree, fw)
		fw.close()
		
	def grabTree(self, filename) :
		fr = open(filename, 'rb')
		return pickle.load(fr)
			


			
		
	
