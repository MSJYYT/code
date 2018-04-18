from math import log
import operator
#本节例子：最终数据结果（第三列）只有两种：鱼类（2个）和非鱼类（3个）
#它们所占比例分别为2/5、3/5，那么H(鱼类判断)=-（2/5log2/5 + 3/5log3/5）
#H(鱼类判断)既为熵
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:#遍历每个实例，统计标签的频次
        currentLabel = featVec[-1]    #取得最后一列数据，计算该属性取值情况有多少个
        if currentLabel not in labelCounts.keys():labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries   #标签的概率分布
        shannonEnt -= prob * log(prob,2)
    return shannonEnt


#定义按照某个特征进行划分的函数splitDataSet
#输入三个变量（待划分的数据集，特征，分类值）
# myDat,labels = creatDataSet()
# splitDataSet(myDat,0,1)
# [[1,'yes'],[1,'yes'],[0,'no']]
# splitDataSet(myDat,0,0)
# [[1,'no'],[1,'no']]
def splitDataSet(dataset,axis,value):
    retDataSet = []
    for featVec in dataset:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0;bestFeature = -1
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]#得到某个特征下所有值（某列）
        uniqueVals = set(featList)#set无重复的属性特征值，得到所有无重复的属性取值
        # 计算每个属性i的概论熵
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet,i,value)#得到i属性下取i属性为value时的集合
            prob = len(subDataSet)/float(len(dataSet))#每个属性取值为value时所占比重
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy#当前属性i的信息增益
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature#返回最大信息增益属性下标
#递归创建树,用于找出出现次数最多的分类名称
def majorityCnt(clasList):
    classCount = {}
    for vote in clasList:#统计当前划分下每中情况的个数
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(),
                              key=operator.itemgetter(1),reverse=True)#reversed=True表示由大到小排序
    return sortedClassCount[0][0]#对字典里的元素按照value值由大到小排序

def createTree(dataSet,labels):
    classList = [example[-1] for example in dataSet]#创建数组存放所有标签值,取dataSet里最后一列（结果）
    # 类别相同，停止划分
    # 判断classList里是否全是一类，count() 方法用于统计某个元素在列表中出现的次数
    if classList.count(classList[0]) == len(classList):
        return classList[0]#当全是一类时停止分割
    if len(dataSet[0]) == 1:#当没有更多特征时停止分割，即分到最后一个特征也没有把数据完全分开，就返回多数的那个结果
        return majorityCnt(classList)
    # 按照信息增益最高选取分类特征属性
    bestFeat = chooseBestFeatureToSplit(dataSet)#返回分类的特征序号,按照最大熵原则进行分类
    bestFeatLabel = labels[bestFeat]#该特征的label, #存储分类特征的标签
    myTree = {bestFeatLabel:{}}#构建树的字典
    del(labels[bestFeat])#从labels的list中删除该label
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]#子集合 ,将labels赋给sublabels，此时的labels已经删掉了用于分类的特征的标签
        # 构建数据的子集合，并进行递归
        myTree[bestFeatLabel][value] = createTree(splitDataSet
                                                  (dataSet,bestFeat,value),subLabels)
    return myTree

