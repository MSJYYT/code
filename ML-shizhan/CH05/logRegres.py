from numpy import *
import matplotlib.pyplot as plt

def loadDataSet():
    dataMat = [];labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0,float(lineArr[0]),float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat,labelMat

def sigmoid(inX):
    return 1.0/(1+exp(-inX))

#梯度上升算法
def gradAscent(dataMatIn, classLabels):
    #一次只用一个样本点更新回归系数w
    dataMatrix = mat(dataMatIn)
    #标签向量转置为列矩阵
    labelMat = mat(classLabels).transpose()
    m,n = shape(dataMatrix)
    alpha = 0.001
    maxCycles = 500
    weights = ones((n,1))
    for k in range(maxCycles):
        h = sigmoid(dataMatrix*weights)
        error = (labelMat - h)
        weights = weights + alpha * dataMatrix.transpose() * error
    return weights

#随机梯度上升算法
def stocGradAscent0(dataMatrix,classLabels):
    #dataMatrix = mat(dataMatIn)
    # labelMat = mat(classLabels).transpose()
    m,n = shape(dataMatrix)
    alpha = 0.01
    weights = ones(n)
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i]*weights))
        error = classLabels[i] - h
        weights = weights + alpha * error * dataMatrix[i]
    return weights

#改进的随机梯度上升算法
def stocGradAscent1(dataMatrix,classLabels,numIter=150):
    #dataMatrix = mat(dataMatrix)
    labelMat = mat(classLabels).transpose()
    m,n = shape(dataMatrix)
    weights = ones(n)
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4/(1.0+j+i)+0.01
            randIndex = int(random.uniform(0,len(dataIndex)))
            h = sigmoid(sum(dataMatrix[randIndex]*weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha *error * dataMatrix[randIndex]
            del(dataIndex[randIndex])
    return weights

#画出数据集和Logistic回归最佳拟合直线的函数
def plotBestFit(weights,weights1,weights2):
    #weights = wei.getA()
    dataMat,labelMat = loadDataSet()
    dataArr = array(dataMat)#矩阵转化为数组
    n = shape(dataArr)[0]
    xcord1 = [];ycord1 = []
    xcord2 = [];ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i,1]);ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1]);ycord2.append(dataArr[i,2])
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    fig = plt.figure(figsize=(10,10),dpi=80)
    #在子图中画出样本点
    ax = fig.add_subplot(221)
    ax.scatter(xcord1,ycord1,s=30,c='red',marker='s')
    ax.scatter(xcord2,ycord2,s=30,c='green')
    #画出拟合直线
    x = arange(-3.0,3.0,0.1)
    y = (-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x,y)
    ax.set_title('梯度上升算法')
    #plt.suptitle('梯度上升算法')
    plt.xlabel('X1');plt.ylabel('X2')
    #plt.show()


    ax2 = fig.add_subplot(222)
    ax2.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax2.scatter(xcord2, ycord2, s=30, c='green')
    # 画出拟合直线
    x2 = arange(-3.0, 3.0, 0.1)
    y2 = (-weights1[0] - weights1[1] * x2) / weights1[2]
    ax2.plot(x2, y2)
    ax2.set_title('随机梯度上升算法')
    #plt.suptitle('随机梯度上升算法')
    plt.xlabel('X1');
    plt.ylabel('X2')
    #plt.show()


    ax3 = fig.add_subplot(223)
    ax3.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax3.scatter(xcord2, ycord2, s=30, c='green')
    # 画出拟合直线
    x3 = arange(-3.0, 3.0, 0.1)
    y3 = (-weights2[0] - weights2[1] * x3) / weights2[2]
    ax3.plot(x3, y3)
    #plt.suptitle('改进随机梯度上升算法')
    ax3.set_title('改进随机梯度上升算法')
    plt.xlabel('X1');
    plt.ylabel('X2')
    plt.show()

dataArr,labelMat = loadDataSet()
weights = gradAscent(dataArr,labelMat)

weights1 = stocGradAscent0(array(dataArr),labelMat)

weights2 = stocGradAscent1(array(dataArr),labelMat)
plotBestFit(array(weights),weights1,weights2)