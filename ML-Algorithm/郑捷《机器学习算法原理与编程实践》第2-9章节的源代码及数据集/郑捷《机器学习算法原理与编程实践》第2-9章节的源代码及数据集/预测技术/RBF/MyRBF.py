import numpy as np
from numpy import *
import matplotlib.pyplot as plt
import sys
import os

# （1）绘制图形函数
def plotscatter(Xmat,Ymat,yHat,plt):
#	print(Xmat)
#	print(Ymat)
	fig = plt.figure()
	ax  = fig.add_subplot(111) #绘制图形位置
	ax.scatter(Xmat.T.tolist(),Ymat.T.tolist(),c='blue',marker='o')#绘制散点图
	plt.plot(Xmat,yHat,'r')
	plt.show()
    
def loadDataSet(filename):     #加载数据集
    numFeat = len(open(filename).readline().split())-1
    X = [];Y = []
    fr = open(filename)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        X.append([float(curLine[i]) for i in range(numFeat)])
        Y.append(float(curLine[-1]))
    return X,Y
    
#局部加权线性回归算法
xArr,yArr = loadDataSet("nolinear.txt") #数据矩阵，分类标签
#RBF函数的平滑系数
miu = 0.02
k   = 0.03
#数据集坐标数组转换矩阵
xMat = mat(xArr)
yMat = mat(yArr).T
testArr = xArr  #测试数组
m,n     = shape(xArr) #xArr的行数
yHat    = zeros(m)    #yHat是y的预测值，yHat的数据是y的回归线矩阵
for i in range(m):                                          #for i in range(m):
	weights = mat(eye(m)) #权重矩阵
	for j in range(m):                                    #for j in range(m):
#		print(testArr[i])
#		print('\n')
#		print(xMat[j,:])
		diffMat = testArr[i]-xMat[j,:]                                                    #list与mat相减，结果是mat类型
		#利用RBF函数计算权重矩阵，计算后的权重是一个对角阵
		weights[j,j] = exp(diffMat*diffMat.T/(-miu*k**2))
	xTx = xMat.T*(weights*xMat)
	if linalg.det(xTx) != 0.0:    #行列式不为0
		ws = xTx.I * (xMat.T * (weights * yMat))
#		ws = linalg.inv(xMat.T*xMat)*(xMat.T*yMat)#矩阵的正规方程组的公式：inv(X.T*X)
		yHat[i] = testArr[i]*ws  #计算回归线坐标矩阵
	else:
		print("This matrix is sigular,cannot do inverse")
		sys.exit(0)#退出程序
plotscatter(xMat[:,1],yMat,yHat,plt)

