import numpy as np
from numpy import *
import matplotlib.pyplot as plt

class PlattSVM(object) :
	def __init__(self) :
		self.X = []                                                                                    #输入的数据集
		self.labelMat = []                                                                     #存储类别标签的数组
		self.C = 0                                                                                    #惩罚因子
		self.tol = 0                                                                                 #容错律
		self.b = 0                                                                                    #截距初始值
		self.kValue = {}                                                                        #liner是线性核函数，Guassian是高斯核函数
		self.maxIter = 100                                                                  #最大迭代次数
		self.svIndex = []                                                                      #支持向量下标
		self.sptVects = []                                                                     #支持向量
		
	def loadDataSet(self, filename) :                                          #加载数据集
		fr = open(filename)
		for line in fr.readlines() :
			lineArr = line.strip().split('\t')
			self.X.append([float(lineArr[0]), float(lineArr[1])])
			self.labelMat.append(float(lineArr[2]))
		self.initparam()                                                                       #导入初始化后需要初始化的参数
		
	def initparam(self) :
		self.X = mat(self.X)                                                                  #数据集矩阵
		self.labelMat = mat(self.labelMat).T                                #类别标签矩阵
		self.m = shape(self.X)[0]                                                       #数据集行数
		self.lambdas = mat(zeros((self.m, 1)))                             #拉格朗日乘子向量
		self.eCache = mat(zeros((self.m, 2)))                                #误差缓存
		self.K = mat(zeros((self.m, self.m)))                                  #存储用于核函数计算的向量
		for i in range(self.m) :
			self.K[:, i] = self.kernels(self.X, self.X[i, :])                     #kValues
			                                                                                                                                                          
	def kernels(self, dataMat, A) :
		m, n = shape(dataMat)
		K = mat(zeros((m, 1)))
		if list(self.kValue.keys())[0] == 'linear' :
			K = dataMat * A.T
		elif list(self.kValue.keys())[0] == 'Gaussian' :
			for j in range(m) :
				deltaRow = dataMat[j, :] - A
				K[j] = deltaRow * deltaRow.T
			K = exp(K / (-1 * self.kValue['Gaussian'] ** 2))
		else :
			raise NameError('无法识别的核函数')
		return K		
		
	#选择lamda2,从缓存中寻找符合KKT条件并具有最大误差的j	
	def chooseJ(self, i, Ei) :
		maxK = -1; maxDeltaE = 0; Ej = 0
		self.eCache[i] = [1, Ei]                                                           #更新误差缓存
		validEcacheList = nonzero(self.eCache[:, 0].A)[0]
		if (len(validEcacheList)) > 1:
			for k in validEcacheList :
				if k == i :
					continue
				Ek = self.calcEk(k)
				deltaE = abs(Ei - Ek)
				if (deltaE > maxDeltaE) :
					maxK = k;maxDeltaE = deltaE; Ej = Ek
			return maxK, Ej
		else :
			j = self.randJ(i)
			Ej = self.calcEk(j)
		return j, Ej
		
	#随机选择一个不等于 i 的 j
	def randJ(self, i) :
		j  = i
		while(j == i) :
			j = int(random.uniform(0, self.m))
		return j
	
	#计算类别误差	
	def calcEk(self, k) :
		return float(multiply(self.lambdas, self.labelMat).T * self.K[:, k] + self.b) - float(self.labelMat[k])
		
	#剪裁lambda2的函数
	def clipLambda(self, aj, H, L) :
		if aj > H :aj = H
		if L > aj : aj = L
		return aj	

	#数据可视化
	def scatterplot(self, plt) :
		fig = plt.figure()
		ax = fig.add_subplot(111)
		for i in range(shape(self.X)[0]) :
			if self.lambdas[i] != 0 :
				ax.scatter(self.X[i, 0], self.X[i, 1], c = 'green', marker = 's')
			elif self.labelMat[i] == 1 :
				ax.scatter(self.X[i, 0], self.X[i, 1], c = 'blue', marker = 'o')
			elif self.labelMat[i] == -1 :
				ax.scatter(self.X[i, 0], self.X[i, 1], c = 'red', marker = 'o')
				
	#主函数     外循环
	def train(self) :
		step = 0
		entireflag = True
		lambdaPairsChanged = 0
		while (step < self.maxIter) and ((lambdaPairsChanged > 0) or (entireflag)) :
			lambdaPairsChanged = 0
			if entireflag :
				for i in range(self.m) :
					lambdaPairsChanged += self.innerLoop(i)
				step += 1
			else :
				nonBoundIs = nonzero((self.lambdas.A > 0) * (self.lambdas.A < self.C))[0]
				for i in nonBoundIs :
					lambdaPairsChanged += self.innerLoop(i)
				step += 1
			if entireflag : entireflag = False                                                                                          #转换标志位：切换到另一种
			elif (lambdaPairsChanged == 0) :entireflag = True
		print(step)
		self.svIndx = nonzero(self.lambdas.A > 0)[0]
		self.sptVects = self.X[self.svIndx]
		self.SVlabel = self.labelMat[self.svIndx]
		
	#主函数    内循环：innerLoop
	def innerLoop(self, i) :
		Ei = self.calcEk(i)
		if ((self.labelMat[i] * Ei < -self.tol) and (self.lambdas[i] < self.C)) or ((self.labelMat[i] * Ei > self.tol) and (self.lambdas[i] > 0)) :
			j, Ej = self.chooseJ(i, Ei)
			lambdaIold = self.lambdas[i].copy()
			lambdaJold = self.lambdas[j].copy()
			if (self.labelMat[i] != self.labelMat[j]) :
				L = max(0, self.lambdas[j] - self.lambdas[i])
				H = min(self.C, self.C + self.lambdas[j] - self.lambdas[i])
			else :
				L = max(0, self.lambdas[j] - self.lambdas[i] - self.C)
				H = min(self.C, self.lambdas[j] - self.lambdas[i])
			if L == H :
				return 0
			eta = 2.0 * self.K[i, j] - self.K[i, i] - self.K[j, j]                                             #公式8.35
			if eta >= 0 :
				return 0
			self.lambdas[j] -=self.labelMat[j] * (Ei - Ej) / eta
			self.lambdas[j] = self.clipLambda(self.lambdas[j], H, L)
			self.eCache[j] = [1, self.calcEk(j)]                                                               #计算个更新 j 的缓存
			if (abs(self.lambdas[j] - lambdaJold < 0.00001)) :
				return 0
			self.lambdas[i] += self.labelMat[j] * self.labelMat[i] * (lambdaJold - self.lambdas[j])
			self.eCache[i] = [1, self.calcEk(i)]                                                              #计算个更新 i 的缓存
			b1 = self.b - Ei - self.labelMat[j] * (self.labelMat[i] - lambdaIold) * self.K[i, i] - self.labelMat[j] * (self.lambdas[j] - lambdaJold) * self.K[i, j]
			b1 = self.b - Ei - self.labelMat[j] * (self.labelMat[i] - lambdaIold) * self.K[i, i] - self.labelMat[j] * (self.lambdas[j] - lambdaJold) * self.K[i, j]
			if (0 < self.lambdas[i]) and (self.C > self.lambdas[i]) :
				self.b = b1
			elif (0 < self.lambdas[j]) and (self.C > self.lambdas[j]) :
				self.b = b2
			else :
				self.b = (b1 + b2) / 2.0
			return 1
		else : return 0

	def classify(self, testSet, testLabel) :
		errorCount = 0
		testMat = mat(testSet)
		m, n = shape(testMat)
		for i in range(m) :
			kernelEval = self.kernels(self.sptVects, testMat[i, :])
			predict = kernelEval.T * multiply(self.SVlabel, self.lambdas[self.svIndx]) + self.b
			if sign(predict) != sign(testLabel[i]) : errorCount += 1
			return float(errorCount) / float(m)

svm = PlattSVM()
svm.C = 100                                                                                   #惩罚因子
svm.tol = 0.001                                                                            #容错律
svm.maxIter = 10000
svm.kValue['Gaussian'] = 3.0                                                  #核函数
svm.loadDataSet('svm.txt')
svm.train()
print(svm.svIndx)
print(shape(svm.sptVects)[0])
print("b: ", svm.b)
svm.scatterplot(plt)
plt.show()
print('\n')
print(svm.classify(svm.X, svm.labelMat))
