from numpy import *
import matplotlib.pyplot as plt

def loadDataSet(filename):
    dataMat = []
    fr = open(filename)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = list(map(float,curLine))
        dataMat.append(fltLine)
    return dataMat

#计算两个向量欧式距离
def distEclud(vecA,vecB):
    return sqrt(sum(power(vecA - vecB,2)))

#为给定数据集构建一个包含k个随机质心的集合
def randCent(dataSet,k):
    n = shape(dataSet)[1]           #列数
    centroids = mat(zeros((k,n)))
    for j in range(n):#一共有J列
        minJ = min(dataSet[:,j])
        rangeJ = float(max(dataSet[:,j]) - minJ)
        centroids[:,j] = minJ + rangeJ * random.rand(k,1)  #生成一个k行1列的矩阵
    return centroids            #返回k行j列的矩阵，即K个质心

def kMeans(dataSet,k,distMeas=distEclud,createCent=randCent):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m,2))) #分配结果矩阵；第1列记录该样本的索引值（即属于第k个质心）
    centroids = createCent(dataSet,k)
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):#第i个样本
            minDist = inf;minIndex = -1
            for j in range(k):#寻找最近的质心；第j个质心
                #第i个样本点距离第j个质心的距离
                distJI = distMeas(centroids[j,:],dataSet[i,:])
                #如果这个距离比最小距离小，找离i样本点最近的质心
                if distJI < minDist:
                    minDist = distJI;minIndex = j
            if clusterAssment[i,0] != minIndex:clusterChanged = True
            clusterAssment[i,:] = minIndex,minDist**2
        print(centroids)
        for cent in range(k):#遍历所有质心并更新他们的取值
            #首先通过数组过滤来获得给定簇的所有点；
            #然后计算所有点的均值，axis=0表示沿矩阵列方向进行均值计算
            ptsInClust = dataSet[nonzero(clusterAssment[:,0].A==cent)[0]]#索引值是cent质心的样本点
            centroids[cent,:] = mean(ptsInClust,axis=0)#计算所有cent质心的样本点的均值，作为新的质心
    return centroids,clusterAssment

#二分K-均值聚类算法
def biKmeans(dataSet,k,distMeas=distEclud):
    m = shape(dataSet)[0]
    #存储数据集中每个点的分配结果及平方误差
    clusterAssment = mat(zeros((m,2)))
    #计算整个数据集的质心，并用一个列表来保留所有质心
    centroid0 = mean(dataSet,axis=0).tolist()[0]
    centList = [centroid0]
    #每个点到质心的误差值
    for j in range(m):
        clusterAssment[j,1] = distMeas(mat(centroid0),dataSet[j,:])**2
    
    while (len(centList) < k):
        lowestSSE = inf
        for i in range(len(centList)):
            ptsInCurrCluster = dataSet[nonzero(clusterAssment[:,0].A==i)[0],:]
            centroidMat,splitClustAss = kMeans(ptsInCurrCluster,2,distMeas)
            sseSplit = sum(splitClustAss[:,1])
            sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:,0].A!=i)[0],1])
            print('sseSplit,and notSplit:',sseSplit,sseNotSplit)
            if (sseSplit + sseNotSplit) < lowestSSE:
                bestCentToSplit = i
                bestNewCents = centroidMat
                bestClustAss = splitClustAss.copy()
                lowestSSE = sseSplit + sseNotSplit
        bestClustAss[nonzero(bestClustAss[:,0].A==1)[0],0] = len(centList)
        bestClustAss[nonzero(bestClustAss[:,0].A==0)[0],0] = bestCentToSplit
        print('the bestCentToSplit is:',bestCentToSplit)
        print('the len of bestClustAss is:',len(bestClustAss))
        centList[bestCentToSplit] = bestNewCents[0,:]
        centList.append(bestNewCents[1,:])
        clusterAssment[nonzero(clusterAssment[:,0].A==bestCentToSplit)[0],:] = bestClustAss
    return mat(centList),clusterAssment

def show(dataSet, k, centroids, clusterAssment):
    #from matplotlib import pyplot as plt
    numSamples, dim = dataSet.shape
    mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']
    for i in range(numSamples):
        markIndex = int(clusterAssment[i, 0])
        plt.plot(dataSet[i, 0], dataSet[i, 1], mark[markIndex])
    mark = ['Dr', 'Db', 'Dg', 'Dk', '^b', '+b', 'sb', 'db', '<b', 'pb']
    for i in range(k):
        plt.plot(centroids[i, 0], centroids[i, 1], mark[i], markersize = 12)
    plt.show()

if __name__ == "__main__":
    dataMat = mat(loadDataSet('testSet.txt'))
    myCentroids, clustAssing = kMeans(dataMat, 4)
    print(myCentroids)
    show(dataMat, 4, myCentroids, clustAssing)



