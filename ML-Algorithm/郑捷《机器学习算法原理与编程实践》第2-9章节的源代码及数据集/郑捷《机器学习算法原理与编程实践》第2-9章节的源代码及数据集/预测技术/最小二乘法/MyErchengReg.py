import numpy as np
from numpy import *
import matplotlib.pyplot as plt


def loadDataSet(filename):     #加载数据集
	X = [];Y = []
	fr = open(filename)
	for line in fr.readlines():
		curLine = line.strip().split('\t')
		X.append(float(curLine[0]))
		Y.append(float(curLine[-1]))
	return X,Y

def plotscatter(Xmat,Ymat,a,b,plt):
	fig = plt.figure()
	ax  = fig.add_subplot(111) #绘制图形位置
	ax.scatter(Xmat,Ymat,c='blue',marker='o')#绘制散点图
	Xmat.sort()                #对Xmat元素进行排序
	yhat = [a*float(xi)+b for xi in Xmat] #计算预测值
	plt.plot(Xmat,yhat,'r')
	plt.show()
	return  yhat

#主函数    
Xmat,Ymat = loadDataSet("regdataset.txt") #导入数据文件
meanX     = mean(Xmat)                      #原始数据的均值
meanY     = mean(Ymat)                      #原始数据的均值
dX        = Xmat-meanX                      #各元素与均值的差
dY        = Ymat-meanY                      #各元素与均值的差
#手工计算
# sumXY = 0;Sqx = 0
# for i in xrange(len(dx)):
#     sumXY += double(dx[i])*double(dy[i])
#     Sqx   = double(dX[i])**2

sumXY = vdot(dX,dY) #返回两个向量的点乘multiply
print(sumXY)
Sqx   = sum(power(dX,2))#向量的平方：（X-meanX)^2

#计算斜率和截距
a = sumXY/Sqx
b = meanY-a*meanX
print (a,b)
#绘制图形
plotscatter(Xmat,Ymat,a,b,plt)

'''
#正规方程组法
#数据矩阵，分类标签
xArr,yArr = loadDataSet("regdataset.txt") #导入数据文件

m = len(xArr) #生成X坐标列
Xmat = mat(ones((m,2)))
for i in range(m):
    Xmat[i,1] = xArr[i]
Ymat = mat(yArr).T   #转化为Y列
xTx = Xmat.T*Xmat

ws = [] #直线的斜率和截距
if linalg.det(xTx) != 0.0:    #行列式不为0
    ws = linalg.inv(Xmat.T*Xmat)*(Xmat.T*Ymat)#矩阵的正规方程组的公式：inv(X.T*X)*X.T*Y
else:
    print  u"矩阵为奇异阵，无逆矩阵"
    sys.exit(0)#退出程序
print  "ws:",ws
'''
