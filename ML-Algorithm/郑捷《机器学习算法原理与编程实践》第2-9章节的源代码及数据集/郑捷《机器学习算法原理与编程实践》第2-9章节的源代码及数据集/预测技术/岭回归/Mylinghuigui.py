from numpy import *
import matplotlib.pyplot as plt

def loadDataSet(filename):
    numFeat    = len(open(filename).readline().split('\t'))-1#get number of fields
    dataMat    = []
    labelMat   = []
    fr         = open(filename)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat
    
#标准化数据集
def normData(xArr,yArr):
    xMat = mat(xArr)
    yMat = mat(yArr).T
    yMean = mean(yMat,0)
    xMean = mean(xMat,0)
    ynorm = yMat - yMean
    xVar  = var(xMat,0)
    xnorm = (xMat-xMean)/xVar
    return xnorm,ynorm
    
def scatterplot(wMat,k):#绘制图形
    fig = plt.figure()
    ax  = fig.add_subplot(111)
    wMatT = wMat.T
    m,n   = shape(wMatT)
    for i in range(m):
        ax.plot(k,wMatT[i,:])
        ax.annotate("feature["+str(i)+"]",xy = (0,wMatT[i,0]),color = 'black')
    plt.show()
    
#前8列为Arr，后1列为yArr
xArr,yArr = loadDataSet('ridgedata.txt')
xMat,yMat = normData(xArr,yArr) #标准化数据集

Knum      = 30 #确定k的迭代次数
wMat      = zeros((Knum,shape(xMat)[1]))
klist     = zeros((Knum,1))
for i in range(Knum):                              #for i in range(Knum):
    k = float(i)/500  #算法的目的是确定k的值
    klist[i] = k      #k值列表
    xTx      = xMat.T*xMat
    m,n = shape(xMat)
    denom    = xTx + eye(shape(xMat)[1])*k
    if linalg.det(denom) == 0.0:
        print ("This matrix is singular,connot do inverse")
        sys.exit(0)
    ws = linalg.inv(denom) * (xMat.T*yMat)
    wMat[i,:] = ws.T
#print(wMat)
#print (klist)
scatterplot(klist,klist)
scatterplot(wMat,klist)
