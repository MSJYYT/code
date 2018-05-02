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
if __name__ == "__main__":
    dataMat,classLabels = loadSimpData()
    classifierArray = adaBoostTrainDS(dataMat,classLabels,9)
