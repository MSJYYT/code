from numpy import *

def loadDataSet(fileName,delim='\t'):
    fr = open(fileName)
    stringArr = [line.strip().split(delim) for line in fr.readlines()]
    datArr = [list(map(float,line))  for line in stringArr]
    return mat(datArr)

#pca特征维度压缩函数
#@dataMat 数据集矩阵
#@topNfeat 需要保留的特征维度，即要压缩成的维度数，默认4096
def pca(dataMat,topNfeat=9999999):
    # 求数据矩阵每一列的均值
    meanVals = mean(dataMat,axis=0)
    # 数据矩阵每一列减去该列的特征均值
    meanRemoved = dataMat - meanVals
    # 计算协方差矩阵，除数n-1是为了得到协方差的无偏估计
    # cov(X,0) = cov(X) 除数是n-1(n为样本个数)
    # cov(X,1) 除数是n
    covMat = cov(meanRemoved,rowvar=0)
    # 计算协方差矩阵的特征值及对应的特征向量
    # 均保存在相应的矩阵中
    eigVals,eigVects = linalg.eig(mat(covMat))
    # sort():对特征值矩阵排序(由小到大)
    # argsort():对特征值矩阵进行由小到大排序，返回对应排序后的索引
    eigValInd = argsort(eigVals)
    # 从排序后的矩阵最后一个开始自下而上选取最大的N个特征值，
    # 返回其对应的索引
    eigValInd = eigValInd[:-(topNfeat+1):-1]
    # 将特征值最大的N个特征值对应索引的特征向量提取出来，组成压缩矩阵
    redEigVects = eigVects[:,eigValInd]
    # 将去除均值后的数据矩阵*压缩矩阵，转换到新的空间，使维度降低为N
    lowDDataMat = meanRemoved * redEigVects
    # 利用降维后的矩阵反构出原数据矩阵
    # (用作测试，可跟未压缩的原矩阵比对)
    reconMat = (lowDDataMat * redEigVects.T) + meanVals
    # 返回压缩后的数据矩阵即该矩阵反构出原始数据矩阵
    return lowDDataMat,reconMat

# dataMat = loadDataSet('testSet.txt')
# lowDMat,reconMat = pca(dataMat,1)
# import matplotlib.pyplot as plt
# fig  = plt.figure()
# ax = fig.add_subplot(111)
# ax.scatter(dataMat[:,0].flatten().A[0],dataMat[:,1].flatten().A[0],
#            marker = '^',s=90)
# ax.scatter(reconMat[:,0].flatten().A[0],reconMat[:,1].flatten().A[0],
#            marker='^',s=50,c='red')
# plt.show()
