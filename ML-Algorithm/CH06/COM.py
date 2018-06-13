from numpy import *
import matplotlib.pyplot as plt

class Kohonen(object):
    def __init__(self):
        self.lratemax = 0.8     #最大学习率--欧式距离
        self.lratemin = 0.05    #最小学习率--欧式距离
        self.rmax = 5.0         #最大聚类半径--根据数据集
        self.rmin = 0.5         #最小聚类半径--根据数据集
        self.Steps = 1000       #迭代次数
        self.lratelist = []     #学习率收敛曲线
        self.rlist = []         #学习半径收敛曲线
        self.w = []             #权重向量组
        self.M = 2              #M×N表示聚类总数
        self.N = 2              #M,N表示邻域的参数
        self.dataMat = []       #外部导入数据集
        self.classLabel = []    #聚类后的类别标签

    def normalize(self,dataMat):        #数据标准化
        [m,n] = shape(dataMat)
        for i in range(n-1):
            dataMat[:,i] = (dataMat[:,i]-mean(dataMat[:,i]))/(std(dataMat[:,i])+1.0e-10)
        return dataMat

    def distEclud(self, matA, matB):        #计算矩阵欧式距离
        ma, na = shape(matA);
        mb, nb = shape(matB);
        rtnmat = zeros((ma, nb))
        for i in range(ma):
            for j in range(nb):
                rtnmat[i, j] = linalg.norm(matA[i, :] - matB[:, j].T)
        return rtnmat

    def loadDataSet(self,fileName):
        numFeat = len(open(fileName).readline().split('\t')) - 1
        fr = open(fileName)
        for line in fr.readlines():
            lineArr = []
            curLine = line.strip().split('\t')
            lineArr.append(float(curLine[0]))
            lineArr.append(float(curLine[1]))
            self.dataMat.append(lineArr)
        self.dataMat = mat(self.dataMat)

     #初始化第二层网络
    def init_grid(self):
        k = 0           #构建第二层网络模型
        grid = mat(zeros((self.M*self.N,2)))
        for i in range(self.M):
            for j in range(self.N):
                grid[k,:] = [i,j]
                k += 1
        return grid

    def ratecalc(self, i):
        lrate = self.lratemax - (i + 1.0) * (self.lratemax - self.lratemin) / self.Steps
        r = self.rmax - ((i + 1.0) * (self.rmax - self.rmin)) / self.Steps
        return lrate, r

    #聚类算法主程序
    def train(self):
        dm,dn = shape(self.dataMat)         #1.构建输入层网络
        normDataset = self.normalize(self.dataMat)
        grid = self.init_grid()             #2.初始化第二层分类网络
        self.w = random.rand(dn,self.M*self.N) #3.随机初始化两层之间的权重向量
        distM = self.distEclud()

        #4.迭代求解
        if self.Steps < 5*dm: self.Steps = 5*dm     #设定最小迭代次数
        for i in range(self.Steps):
            lrate,r = self.ratecalc(i)  #1)计算当前迭代次数下的学习率和分类半径
            self.lratelist.append(lrate)
            self.rlist.append(r)
            #2)随机生成样本索引，并抽取一个样本
            k = random.randint(0,dm)
            mySample = normDataset[k,:]
            #3)计算最优节点：返回最小距离的索引值
            minIndx = (distM(mySample,self.w)).argmin()
            #4)计算邻域
            d1 = ceil(minIndx/self.M)#计算此节点在第二层矩阵中的位置
            d2 = mod(minIndx,self.M)
            distMat = distM(mat([d1,d2]),grid.T)
            nodelindx = (distMat<r).nonzero()[1]    #获取邻域内所有节点
            for j in range(shape(self.w)[1]):   #5)按列更新权重
                if sum(nodelindx == j):
                    self.w[:,j] = self.w[:,j]+ lrate*(mySample[0]-self.w[:,j])
        #主循环结束
        #分配和存储聚类后的类别标签
        self.classLabel = range(dm)
        for i in range(dm):
            self.classLabel[i] = distM(normDataset[i,:],self.w).argmin()
        self.classLabel = mat(self.classLabel)

    def showCluster(self,plt):
        lst = unique(self.classLabel.tolist()[0])   #去重
        i = 0
        for cindx in lst:
            myclass = nonzero(self.classLabel == cindx)[1]
            xx = self.dataMat[myclass].copy()
            if i == 0:      plt.plot(xx[:,0],xx[:,1],'bo')
            elif i == 1:    plt.plot(xx[:,0],xx[:,1],'rd')
            elif i == 2:    plt.plot(xx[:, 0], xx[:, 1], 'gD')
            elif i == 3:    plt.plot(xx[:, 0], xx[:, 1], 'c^')
        plt.show()