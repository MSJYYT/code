import sys
import os
from numpy import *
import numpy as np
import matplotlib.pyplot as plt

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
    #plt.scatter(mydata.T[0].tolist(),mydata.T[1].tolist(),s = size,c = color,marker = mrkr)
	plt.scatter(mydata.T[0].tolist(),mydata.T[1].tolist(),s = size,c = color,marker = mrkr)

def distEclud(vecA,vecB):
    return linalg.norm(vecA-vecB)

#随机生成聚类中心
def randCenters(dataSet,k):
	n = shape(dataSet)[1]               
	clustercents = mat(zeros([k,n]))
	for col in range(n):
		mincol = min(dataSet[:,col])
		mincol = float(mincol.tolist()[0][0])         #重点难点，由[['1.7841']]变成float类型的1.7841
		maxcol = max(dataSet[:,col])
		maxcol = float(maxcol.tolist()[0][0])
		clustercents[:,col] = mat(mincol+(maxcol-mincol)*random.rand(k,1))
	return clustercents

#kMean算法
def kMean(dataSet, k) :
	m = dataSet.shape[0]                                               #m = 400
	ClusDist = mat(zeros((m,2))) 
	clustercents = randCenters(dataSet,k)   
	flag = True                                                    
	counter = []
	while flag:                               
		flag = False      
		for i in range(m) :
			distlist = [distEclud(clustercents[j,:],dataSet[i,:]) for j in range(k)]
			minDist  = min(distlist)
			minIndex = distlist.index(minDist)
			if ClusDist[i,0] != minIndex :
				flag = True              
			ClusDist[i,:] = minIndex,minDist
		for cent in range(k) :
			pstInClust = dataSet[nonzero(ClusDist[:,0].A == cent)[0]]
			clustercents[cent,:] = mean(pstInClust,axis = 0)
	return clustercents, ClusDist
'''	
#主函数
dataMat = file2matrix("4k2_far.txt","\t")
dataSet = mat(dataMat[:,1:])             
k = 4
clustercents, ClusDist = kMean(dataSet, k)
print("clustercents:\n",clustercents)
color_cluster(ClusDist[:,0:1],dataSet,plt)
drawScatter(plt,clustercents,size=60,color='red',mrkr='D')
plt.show()
'''

