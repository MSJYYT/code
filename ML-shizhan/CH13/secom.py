from CH13 import pca
from numpy import *
import matplotlib.pyplot as plt

#缺失值处理函数
def replaceNaNWithMean():
    datMat = pca.loadDataSet('secom.data',' ')
    # 获取特征维度
    numFeat = shape(datMat)[1]
    # 遍历数据集每一个维度
    for i in range(numFeat):
        # 利用该维度所有非NaN特征求取均值
        meanVal = mean(datMat[nonzero(~isnan(datMat[:,i].A))[0],i])
        # 将该维度中所有NaN特征全部用均值替换
        datMat[nonzero(isnan(datMat[:,i].A))[0],i] = meanVal
    return datMat

dataMat = replaceNaNWithMean()
meanVals = mean(dataMat,axis=0)
meanRemoved = dataMat - meanVals
covMat = cov(meanRemoved,rowvar=0)
eigVals,eigVects = linalg.eig(mat(covMat))
print(eigVals)
