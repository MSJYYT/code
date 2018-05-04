from numpy import *
from CH07 import boost

def loadSimpData():
    datMat = matrix([[1,2.1],
                     [2,1.1],
                     [1.3,1],
                     [1,1],
                     [2,1]])
    classLabels = [1,1,-1,-1,1]
    return datMat,classLabels

def adaBoostTrainDS(dataArr,classLabels,numIt=40):
    weakClassArr = []
    m = shape(dataArr)[0]
    D = mat(ones((m,1))/m)
    aggClassEst = mat(zeros((m,1)))
    for i in range(numIt):
        bestStump,error,classEst = boost.buildStump(dataArr,classLabels,D)
        print('D:',D.T)
        alpha = float(0.5*log((1-error)/max(error,1e-16)))
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)
        print('classEst:',classEst.T)
        expon = multiply(-1*alpha*mat(classLabels).T,classEst)
        D = multiply(D,exp(expon))
        D = D/D.sum()
        aggClassEst += alpha*classEst
        print('aggClassEst:',aggClassEst.T)
        aggErrors = multiply(sign(aggClassEst) != mat(classLabels).T,ones((m,1)))
        errorRate = aggErrors.sum()/m
        print('total error:',errorRate)
        if errorRate == 0:break
    return weakClassArr
def adaClassify(datToClass,classifierArr):
    dataMatrix = mat(datToClass)
    m = shape(dataMatrix)[0]
    aggClassEst = mat(zeros((m,1)))
    for i in range(len(classifierArr)):
        classEst = boost.stumpClassify(dataMatrix,classifierArr[i]['dim'],
                                       classifierArr[i]['thresh'],
                                       classifierArr[i]['ineq'])
        aggClassEst += classifierArr[i]['alpha']*classEst
        #print(aggClassEst)
    return sign(aggClassEst)

def loadDataSet(filename):
    numFeat = len(open(filename).readline().split('\t'))
    dataMat = [];labelMat = []
    fr = open(filename)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat-1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat

if __name__ == "__main__":
    # dataMat,classLabels = loadSimpData()
    # classifierArray = adaBoostTrainDS(dataMat,classLabels,9)
    # print(adaClassify([[5,5],[0,0]],classifierArray))
    dataArr,labelArr = loadDataSet('horseColicTraining2.txt')
    classifierArray = adaBoostTrainDS(dataArr,labelArr,10)
    testArr,testLabelArr = loadDataSet('horseColicTest2.txt')
    prediction10 = adaClassify(testArr,classifierArray)
    errArr = mat(ones((67,1)))
    print(prediction10)
    print(errArr[prediction10 != mat(testLabelArr).T].sum())
