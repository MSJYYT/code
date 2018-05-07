from numpy import *

def loadDataSet(filename):
    dataMat = []
    fr = open(filename)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = map(float,curLine)
        dataMat.append(fltLine)
    return dataMat

#计算两个向量欧式距离
def distEclud(vecA,vecB):
    return sqrt(sum(power(vecA - vecB,2)))

def randCent(dataSet,k):
    n = shape(dataSet)[1]           #列数
    centroids = mat(zeros((k,n)))
    for j in range(n):#一共有J列
        minJ = min(dataSet[:,j])
        rangeJ = float(max(dataSet[:,j]) - minJ)
        centroids[:,j] = minJ + rangeJ * random.rand(k,1)  #生成一个k行1列的矩阵
    return centroids            #返回k行j列的矩阵，即K个样本点

if __name__ == "__main__":
    # teMat = mat(eye(4))
    # print(shape(teMat)[1])
    print(random.rand(2,1))



