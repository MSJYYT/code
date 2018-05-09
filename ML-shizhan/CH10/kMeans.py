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

    while (len(centList) < k):#该循环会不停对簇进行划分，直到得到想要的簇数目为止，为此需要比较划分前后的sse
        lowestSSE = inf
        for i in range(len(centList)):#遍历簇列表centList中的每个簇来决定最佳的簇进行划分
            # 对每个簇，对该簇中的所有点看成一个小的数据集ptsInCurrCluster
            #.A将矩阵转为数组
            #表示（clusterAssment第0列等于i的元素坐标，取行坐标）这些坐标映射到dataset中这些行（所有列）
            ptsInCurrCluster = dataSet[nonzero(clusterAssment[:,0].A==i)[0],:]
            # 将ptsInCurrCluster输入到函数kmeans()（k=2）中进行处理生成2个质心簇，并给出每个簇的误差值
            centroidMat,splitClustAss = kMeans(ptsInCurrCluster,2,distMeas)

            # 误差与剩余数据集的误差之和将作为本次划分的误差
            sseSplit = sum(splitClustAss[:,1])
            sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:,0].A!=i)[0],1])
            print('sseSplit,and notSplit:',sseSplit,sseNotSplit)
            # 如果该划分的sse值最小，则本次划分保存
            if (sseSplit + sseNotSplit) < lowestSSE:
                #一旦决定了要划分的簇，就要执行实际划分操作，
                # 即将要划分的簇中所有点的簇分配结果进行修改即可。
                # 当使用KMEANS()函数并簇数为2时，得到两个编号0与1的结果簇，
                # 需要将这些簇编号修改改为划分簇与新加簇的编号，
                # 该过程通过2个数组过滤器完成
                bestCentToSplit = i
                bestNewCents = centroidMat#新划分出的质心
                bestClustAss = splitClustAss.copy()
                lowestSSE = sseSplit + sseNotSplit
        bestClustAss[nonzero(bestClustAss[:,0].A==1)[0],0] = len(centList)
        bestClustAss[nonzero(bestClustAss[:,0].A==0)[0],0] = bestCentToSplit
        print('the bestCentToSplit is:',bestCentToSplit)
        print('the len of bestClustAss is:',len(bestClustAss))
        # 新的簇分配结果被更新
        centList[bestCentToSplit] = bestNewCents[0,:].tolist()[0]
        # 新的质心添加到centlist中
        centList.append(bestNewCents[1,:].tolist()[0])
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
    dataMat = mat(loadDataSet('testSet2.txt'))
    myCentroids, clustAssing = biKmeans(dataMat, 3)
    print(myCentroids)
    show(dataMat, 3, myCentroids, clustAssing)
    # dataMat = mat(loadDataSet('testSet2.txt'))
    # myCentroids, clustAssing = biKmeans(dataMat, 3)
    # print(myCentroids)
    # show(dataMat, 3, myCentroids, clustAssing)




