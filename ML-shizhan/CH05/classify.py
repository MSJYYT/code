from CH05 import logRegres
from numpy import *
#以特征向量和回归系数作为输入来计算对应的sigmoid值
#如果sigmoid值大于0.5函数返回1，否则返回0
def classifyVector(inX, weights):
    prob = logRegres.sigmoid(sum(inX*weights))
    if prob > 0.5 : return 1.0
    else:return 0.0

def colicTest():
    #打开训练集和测试集，最后一列为类别标签
    #数据最初有三个类别标签:"扔存活"“已经死亡”“已经安乐死”
    #将标签“已经死亡”“已经安乐死”合并为“未能存活”
    frTrain = open('horseColicTraining.txt')
    frTest = open('horseColicTest.txt')
    trainingSet = [];trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):

            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
        #计算回归系数向量
    trainWeights = logRegres.stocGradAscent1(array(trainingSet),trainingLabels,500)
    errorCount = 0;numTestVec = 0.0
    #导入测试集并计算分类错误率
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(array(lineArr),trainWeights)) != int(currLine[21]):
            errorCount += 1
    errorRate = (float(errorCount)/numTestVec)
    print("the error rate of this test is:%f" % errorRate)
    return errorRate

def multiTest():
    numTests = 10;errorSum = 0.0
    for k in range(numTests):
        errorSum += colicTest()
    print("after %d iterations the average error rate is:%f" % (numTests,errorSum/float(numTests)))

if __name__ == "__main__":
    multiTest()