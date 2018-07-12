import random as randm
from numpy import *
import copy
import matplotlib.pyplot as plt

class BoltzmannNet(object):
	def __init__(self):
		self.dataMat   = []                              #外部导入的数据集
		self.MAX_ITER  = 2000                    #外循环迭代次数
		self.T0 = 1000                                      #最高温度：这个温度变化范围应位于最大迭代范围
		self.Lambda    = 0.97                        #温度下降参数
		self.iteration = 0                                #达到最优时的迭代次数
		self.dist      = []                                     #存储生成的距离
		self.pathindx  = []                              #存储生成的路径索引
		self.bestdist  = 0                                #生成的最优路径长度
		self.bestpath  = []                              #生成的最优路径

	def loadDataSet(self,filename):       #加载数据集
		self.dataMat     = []
		self.classLabels = []
		fr               = open(filename)
		for line in fr.readlines():
			lineArr = line.strip().split()
			self.dataMat.append([float(lineArr[0]),float(lineArr[1])])
#			self.classLabels.append(int(float(lineArr[2])))
		self.dataMat = mat(self.dataMat)
		m,n          = shape(self.dataMat)
		self.nSampNum = m                        #样本数量
		self.nSampDim = n                           #样本维度

	def distEclud(self,vecA,vecB):
		#欧式距离
		eps = 1.0e-6
		return linalg.norm(vecA-vecB) + eps

	def boltzmann(self,new1,old1,T):
		return exp(-(new1-old1)/T)

	def pathLen(self,dist,path):
		N = len(path)
#		print(list(range(0,N-1)))
		plen = 0
		for i in range(0,N-1):   #长度为N的向量，包含1~N的整数
			plen += dist[path[i],path[i+1]]
		plen += dist[path[0],path[N-1]]
		return  plen

	def changePath(self,old_path):
		N = len(old_path)
		if random.rand() < 0.25:   #产生两个位置，并交换
			chpos = floor(random.rand(1,2)*N).tolist()[0]
			new_path = copy.deepcopy(old_path)
			new_path[int(chpos[0])] = old_path[int(chpos[1])]
			new_path[int(chpos[1])] = old_path[int(chpos[0])]
		else:
			d = ceil(random.rand(1,3)*N).tolist()[0]
			d.sort()  #随机路径排序
			a = int(d[0])
			b = int(d[1])
			c = int(d[2])
			if a != b and b != c :
				new_path = copy.deepcopy(old_path)
				new_path[a:c-1] = old_path[b-1:c-1]+old_path[a:b-1]
			else:
				new_path = self.changePath(old_path)
		return new_path

	def drawPath(self,bestpath):

		plt.plot(self.dataMat[bestpath,0],self.dataMat[bestpath,1],color='red')

	def drawScatter(self,plt):
		px = (self.dataMat[:,0]).tolist()
		py = (self.dataMat[:,1]).tolist()
		plt.scatter(px,py,c = 'green',marker = 'o',s = 60)
		i = 65
		for x,y in zip(px,py):
			plt.annotate(str(chr(i)),xy = (x[0]+40,y[0]),color = 'black')
			i += 1

	def Trendline(self,plt,color = 'b'):
		plt.plot(range(len(self.dist)),self.dist,color)

	def initBMNet(self,m,n,distMat):
		self.pathindx = list(range(m))
		randm.shuffle(self.pathindx)  #随机生成每个路径
		self.dist.append(self.pathLen(distMat,self.pathindx)) #每个路径对应的距离
		return self.T0,self.pathindx,m

	#第一步：导入数据，并根据配置参数初始化网络
	def train(self):  #主函数
		[m,n] = shape(self.dataMat)
		distMat = mat(zeros((m, m)))
		for i in range(m) :
			for j in range(m) :
				distMat[i, j] = self.distEclud(self.dataMat[i, :], self.dataMat[j, :]) #转换为邻接矩阵（距离矩阵）
#		print(distMat)
		#distMat = self.distEclud(self.dataMat,self.dataMat.T) #转换为邻接矩阵（距离矩阵）
		# T为当前温度，curpath为当前路径索引，MAX_M为内循环最大迭代次数
		[T,curpath,MAX_M] = self.initBMNet(m,n,distMat)

		#进入主循环，计算最优路径
		step = 0 #初始化外循环迭代
		while step <= self.MAX_ITER:    #外循环while step <= self.MAX_ITER:
			m = 0                       #内循环计数器
			while m <= MAX_M :          #内循环MAX_M
				curdist = self.pathLen(distMat,curpath) #计算当前路径距离
				newpath = self.changePath(curpath)      #交换产生新路径
				newdist = self.pathLen(distMat,newpath) #计算新路径距离
				#判断网络是否是一个局部稳态
				if (curdist > newdist):
					curpath = newpath
					self.pathindx.append(curpath)
					self.dist.append(newdist)
					self.iteration += 1       #增加迭代次数
				else:                         #如果网络处于局部稳定状态，则执行玻尔兹曼函数
					if random.rand()<self.boltzmann(newdist,curdist,T):
						curpath = newpath
						self.pathindx.append(curpath)
						self.dist.append(newdist)
						self.iteration += 1   #增加迭代次数
				m += 1
			step += 1
			T = T *self.Lambda   #降温，返回迭代，直至算法终止
		#第三步：提取最优路径
		self.bestdist = min(self.dist)
		indxes = argmin(self.dist)
		self.bestpath = self.pathindx[indxes]
if __name__ == '__main__':
    bmNet = BoltzmannNet()
    bmNet.loadDataSet("dataSet25.txt")
    bmNet.train()
    print('循环迭代',bmNet.iteration,'次')
    print('最优解',bmNet.bestdist)
    print('最佳路线',bmNet.bestpath)
    bmNet.drawScatter(plt)
    bmNet.drawPath(bmNet.bestpath)
    plt.show()

    bmNet.Trendline(plt)
    plt.show()



