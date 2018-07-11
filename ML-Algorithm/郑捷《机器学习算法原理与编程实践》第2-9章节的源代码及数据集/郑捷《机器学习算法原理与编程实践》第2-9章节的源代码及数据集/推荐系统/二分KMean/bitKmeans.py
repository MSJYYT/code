import sys
import os
from numpy import *
import numpy as np
import matplotlib.pyplot as plt
from kMean import *
import copy

#文件读取注意点   不含b的读取方法
def file2matrix(path,delimiter):                                
	recordlist = []
	fp = open(path,"r")                                                #这里是“r”，不可以写成"rb"                                                                                                              
	content    = fp.read()
	fp.close()
	rowlist    = content.splitlines()                                                                              
	recordlist = [row.split(delimiter) for row in rowlist if row.strip()]   #这里不需要encode()
	m,n = shape(recordlist)                                                                           #shape适用于list、mat
	myData = np.zeros([m, n])                                    #<class 'numpy.ndarray'>
#	myData = mat(zeros([m, n]))                                 #<class 'numpy.matrixlib.defmatrix.matrix'>
	for i in range(m) :
		for j in range(n) :
			myData[i][j] = float(recordlist[i][j])
#			myData[i,j] = float(recordlist[i][j])                   #第二种访问方式，有警告
	return myData          
	                                                                                   
def color_cluster(dataindx,dataSet,plt,k = 4):
	index    = 0
	datalen  = len(dataindx)
	for indx in range(datalen):                                 #   for indx in range(datalen):
		if int(dataindx[indx]) == 0:
			plt.scatter(dataSet[indx,0],dataSet[indx,1],c='blue',marker='o')
		elif int(dataindx[indx]) == 1:
			plt.scatter(dataSet[indx,0],dataSet[indx,1],c='green',marker='o')
		elif int(dataindx[indx]) == 2:
			plt.scatter(dataSet[indx,0],dataSet[indx,1],c='red',marker='o')
		elif int(dataindx[indx]) == 3:
			plt.scatter(dataSet[indx,0],dataSet[indx,1],c='cyan',marker='o')
		index += 1

def drawScatter(plt,mydata,size = 20,color = 'blue',mrkr = 'o'):
	#print(mydata.T[0])
    #plt.scatter(mydata.T[0].tolist(),mydata.T[1].tolist(),s = size,c = color,marker = mrkr)
	plt.scatter(mydata.T[0].tolist(),mydata.T[1].tolist(),s = size,c = color,marker = mrkr)
	
def distEclud(vecA,vecB):
    return linalg.norm(vecA-vecB)
    
dataMat = file2matrix("4k2_far.txt","\t")
dataSet = mat(dataMat[:,1:])             
m = dataSet.shape[0]                                                                                                #m = 400
k = 4
centroid0 = mean(dataSet, axis = 0).tolist()[0]                                                  #初始化第一个聚类中心：每一列的均值
centList = [centroid0]
ClustDist = mat(zeros([m, 2]))                                                                                  #400*2
#print(len(centList))                                                                                                    #1

for j in range(m):
	ClustDist[j,1] = distEclud(centroid0, dataSet[j,:])**2
while (len(centList) < k):
	lowestSSE = inf                                                                                                                            #无穷大
	for i in range(len(centList)) :
		ptsInCurrCluster = dataSet[nonzero(ClustDist[:,0].A==i)[0],:]
		centroidMat, splitClustAss = kMean(ptsInCurrCluster, 2)
		sseSplit = sum(splitClustAss[:,1])
#		print(len(splitClustAss[:,1]))
		sseNotSplit = sum(ClustDist[nonzero(ClustDist[:,0].A!=i)[0],1])                       #0, 257.6, 103.6
		if (sseSplit + sseNotSplit) < lowestSSE :
			bestCentToSplit = i                                                                                                       #0, 0, 1 聚类中心的最优分割点
			bestNewCents = centroidMat                                                                                  #更新最优聚类中心
			bestClustAss = copy.deepcopy(splitClustAss)                                                    #拷贝聚类距离表为最优聚类距离表
			lowestSSE = sseSplit + sseNotSplit
	bestClustAss[nonzero(bestClustAss[:,0].A == 1)[0],0] = len(centList)             #二分后标签更新
	bestClustAss[nonzero(bestClustAss[:,0].A == 0)[0],0] = bestCentToSplit
	centList[bestCentToSplit] = bestNewCents[0,:].tolist()[0]                                  #加入聚类中心
	centList.append(bestNewCents[1,:].tolist()[0])
	ClustDist[nonzero(ClustDist[:,0].A == bestCentToSplit)[0],:]= bestClustAss #更新SSE的值(sum of squared errors)

color_cluster(ClustDist[:,0:1],dataSet,plt)
print(mat(centList))
drawScatter(plt,mat(centList),size=60,color='red',mrkr='D')
plt.show()

