from numpy import *

def loadDataSet(filename):
    numFeat = len(open(filename).readline().split('\t')) - 1
    dataMat = [];labelMat = []
    fr = open(filename)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat

def standRegres(xArr,yArr):
    xMat = mat(xArr);yMat = mat(yArr).T
    xTx = xMat.T*xMat
    if linalg.det(xTx) == 0:
        print('This matrix is singular,cannot do inverse')
        return
    ws = xTx.I * (xMat.T*yMat)
    return ws

xArr,yArr = loadDataSet('ex0.txt')
ws = standRegres(xArr,yArr)
xMat = mat(xArr)
yMat = mat(yArr)
#yHat = xMat*ws

import matplotlib.pyplot as plt
#绘制原始数据
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(xMat[:,1].flatten().A[0],yMat.T[:,0].flatten().A[0])
#绘制yHat的值
xCopy = xMat.copy()
xCopy.sort(0)
yHat = xCopy*ws
ax.plot(xCopy[:,1],yHat)
plt.show()
print(corrcoef(yHat.T,yMat))