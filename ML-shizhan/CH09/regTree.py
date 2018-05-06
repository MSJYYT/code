class treeNode():
    def __init__(self,feat,val,right,left):
        featureToSplitOn = feat
        valueOfSplit = val
        rightBranch = right
        leftBranch = left

#CART算法
from numpy import *

def loadDataSet(filename):
    dataMat = []
    fr = open(filename)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = list(map(float,curLine))
        dataMat.append(fltLine)
    return dataMat

def plotBestFit(file):              #画出数据集
    import matplotlib.pyplot as plt
    dataMat=loadDataSet(file)       #数据矩阵和标签向量
    dataArr = array(dataMat)        #转换成数组
    n = shape(dataArr)[0]
    xcord1 = []; ycord1 = []        #声明两个不同颜色的点的坐标
    #xcord2 = []; ycord2 = []
    for i in range(n):
        xcord1.append(dataArr[i,0]); ycord1.append(dataArr[i,1])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='green', marker='s')
    #ax.scatter(xcord2, ycord2, s=30, c='green')
    plt.xlabel('X1'); plt.ylabel('X2');
    plt.show()


#输入：数据集合，待切分的特征，该特征的某个值
#用矩阵的语言描述：数据集合的第feature列大于或小于value的行为mat0或mat1
def binSplitDataSet(dataSet,feature,value):
    mat0 = dataSet[nonzero(dataSet[:,feature] > value)[0],:]
    mat1 = dataSet[nonzero(dataSet[:,feature] <= value)[0],:]
    return mat0,mat1

def regLeaf(dataSet):
    '''负责生成叶节点'''
    # 当chooseBestSplit()函数确定不再对数据进行切分时，将调用本函数来得到叶节点的模型。
    # 在回归树中，该模型其实就是目标变量的均值。
    return mean(dataSet[:,-1])
def regErr(dataSet):
    #误差估计函数，该函数在给定的数据上计算目标变量的平方误差，这里直接调用均方差函数
    return var(dataSet[:,-1]) * shape(dataSet)[0]

#最佳二元切分方式
def chooseBestSplit(dataSet,leafType=regLeaf,errType=regErr,ops=(1,4)):
    tolS = ops[0];tolN = ops[1]     #tolS是容许的误差下降值，tolN是切分点最少样本数
    #如果剩余特征值的数量等于1，不需要再切分直接返回，（退出条件1）
    if len(set(dataSet[:,-1].T.tolist()[0])) == 1:
        return None,leafType(dataSet)   #如果所有值都相等则退出
    m,n = shape(dataSet)
    S = errType(dataSet)        #计算平方误差
    bestS = inf;bestIndex = 0;bestValue = 0
    for featIndex in range(n-1):
        temp = dataSet[:,featIndex].tolist()
        #循环整个集合
        for splitVal in set([a[0]for a in temp]):#每次返回的集合中，元素的顺序都将不一样
            mat0,mat1 = binSplitDataSet(dataSet,featIndex,splitVal)#将数据集合切分得到两个子集
            #如果划分的集合的大小小于切分的最少样本数，重新划分
            if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN): continue
            newS = errType(mat0) + errType(mat1)#计算两个集合的平方误差和
            if newS < bestS:
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS
    # 在循环了整个集合后，如果误差减少量(S - bestS)小于容许的误差下降值，则退出，（退出条件2）
    if (S - bestS) < tolS:      #如果误差减少不大则退出
        return None,leafType(dataSet)
    mat0,mat1 = binSplitDataSet(dataSet,bestIndex,bestValue)#按照保存的最佳分割来划分集合
    # 如果切分出的数据集小于切分的最少样本数，则退出，（退出条件3）
    if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):  #如果切分出的数据集很小则退出
        return None,leafType(dataSet)
    return bestIndex,bestValue


def createTree(dataSet,leafType=regLeaf,errType=regErr,ops=(1,4)):
    feat,val = chooseBestSplit(dataSet,leafType,errType,ops)
    if feat == None:return val
    retTree = {}
    retTree['spInd'] = feat
    retTree['spVal'] = val
    lSet,rSet = binSplitDataSet(dataSet,feat,val)
    retTree['left'] = createTree(lSet,leafType,errType,ops)
    retTree['right'] = createTree(rSet,leafType,errType,ops)
    return retTree

if __name__ == '__main__':
    # myDat = loadDataSet('ex00.txt')
    # myMat = mat(myDat)
    # #print(myMat.T)
    # # Tree = createTree(myMat)
    # # print(Tree)
    # plotBestFit('ex00.txt')
    testMat = mat(eye(4))
    print(mean(testMat))
