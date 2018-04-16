from numpy import *
import operator
from os import listdir

def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group,labels

def classify0(inX,dataSet,labels,k):
    """
    inX 是输入的测试样本，是一个[x, y]样式的
    dataset 是训练样本集
    labels 是训练样本标签
    k 是top k最相近的;即与k个距离最近的样本点做投票对比
    """
    # shape返回矩阵的[行数，列数]，
    # 那么shape[0]获取数据集的行数，
    # 行数就是样本的数量
    dataSetSize = dataSet.shape[0]

    # diffMat就是输入样本与每个训练样本的差值，然后对其每个x和y的差值进行平方运算。
    # diffMat是一个矩阵，矩阵**2表示对矩阵中的每个元素进行**2操作，即平方。

    diffMat = tile(inX,(dataSetSize,1))- dataSet
    sqDiffMat = diffMat**2

    # axis=1表示按照横轴，sum表示累加，即按照行进行累加。
    # sqDistance = [[1.01],
    #               [1.0 ],
    #               [1.0 ],
    #               [0.81]]

    sqDistances = sqDiffMat.sum(axis=1)
    # 对平方和进行开根号
    distaces = sqDistances**0.5
    # 按照升序进行快速排序，返回的是原数组的下标。
    sortedDistIndicies = distaces.argsort()
    classCount = {}
    for i in range(k):
        # index = sortedDistIndicies[i]是第i个最相近的样本下标
        # voteIlabel = labels[index]是样本index对应的分类结果('A' or 'B')

        voteIlabel = labels[sortedDistIndicies[i]]

        # classCount.get(voteIlabel, 0)返回voteIlabel的值，如果不存在，则返回0
        # 然后将票数增1

        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    sortedClassCount = sorted(classCount.items(),
                              key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

def label(votelabel):
    if votelabel == "largeDoses":
        return 3
    elif votelabel == "smallDoses":
        return 2
    elif votelabel == "didntLike":
        return 1

def file2matrix(filename):
    # 从文件中读取训练数据，并存储为矩阵
    fr = open(filename)
    numberOfLines = len(fr.readlines()) #获取 n=样本的行数

    returnMat = zeros((numberOfLines,3))
    #创建一个2维矩阵用于存放训练样本数据， 一共有n行，每一行存放3个数据即n行3列

    classLabelVector = []   #创建一个1维数组用于存放训练样本标签。
    fr = open(filename)
    index = 0
    for line in fr.readlines():
        # 把回车符号给去掉
        line = line.strip()
        # 把每一行数据用\t分割
        listFromLine = line.split('\t')
        # 把分割好的数据放至数据集，其中index是该样本数据的下标，就是放到第几行
        returnMat[index,:] = listFromLine[0:3]
        # 把该样本对应的标签放至标签集，顺序与样本集对应。
        classLabelVector.append(label(listFromLine[-1]))
        index += 1
    return returnMat,classLabelVector

def autoNorm(dataSet):
    #训练数据归一化
    # 获取数据集中每一列的最小数值
    minVals = dataSet.min(0)
    # 获取数据集中每一列的最大数值
    maxVals = dataSet.max(0)
    # 最大值与最小的差值
    ranges = maxVals - minVals
    # 创建一个与dataSet同shape的全0矩阵，用于存放归一化后的数据
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    # 把最小值扩充为与dataSet同shape，然后作差
    normDataSet = dataSet - tile(minVals,(m,1))
    # 把最大最小差值扩充为dataSet同shape，然后作商，是指对应元素进行除法运算，
    # 而不是矩阵除法。
    # 矩阵除法在numpy中要用linalg.solve(A,B)
    normDataSet = normDataSet/tile(ranges,(m,1))
    return normDataSet,ranges,minVals

def img2vector(filename):
    #每个样本文件是32行乘32列的数字，共1024个数字
    returnVect = zeros((1,1024))
    fr = open(filename)
    #循环读取32行32列
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect






