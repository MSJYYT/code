from numpy import *
import matplotlib as plt
import random as randm
import copy
class BoltzmannNet(object):
    def __init__(self):
        self.dataMat=[]         #外部导入的数据集
        self.MAX_ITER=2000      #外循环迭代次数
        self.T0=1000            #最高温度，这个温度范围位于最大迭代范围内
        self.Lambda=0.97        #温度下降参数
        self.iteration=0        #达到最优时迭代次数
        self.dist=[]            #存储生成的距离
        self.pathindx=[]        #存储生成的路径索引
        self.bestdist=[]        #生成的最优路径长度
        self.bestpath=[]        #生成的最优路径

    def loadDataset(self,fileName):
        numFeat = len(open(fileName).readline().split('\t')) - 1
        fr = open(fileName)
        for line in fr.readlines():
            lineArr = []
            curLine = line.strip().split('\t')
            lineArr.append(float(curLine[0]))
            lineArr.append(float(curLine[1]))
            self.dataMat.append(lineArr)
        self.dataMat = mat(self.dataMat)

    def distEclud(self,matA,matB):
        ma, na = shape(matA)
        mb, nb = shape(matB)
        rtnmat = zeros((ma, nb))
        for i in range(ma):
            for j in range(nb):
                rtnmat[i, j] = linalg.norm(matA[i, :] - matB[:, j].T)
        return rtnmat

    #玻尔兹曼函数
    def boltzmann(self,newl,oldl,T):
        return exp(-(newl-oldl)/T)

    #计算路径长度
    def pathLen(self,dist,path):
        N=len(path)
        plen=0
        for i in range(0,N-1):
            plen += dist[path[i],path[i+1]]
        plen += dist[path[0],path[N-1]]
        return plen

    #交换新旧路径
    def changePath(self,old_path):
        N=len(old_path)
        if random.rand()<0.25:
            chpos = floor(random.rand(1,2)*N).tolist()[0]
            new_path = copy.deepcopy(old_path)
            new_path[int(chpos[0])] = old_path[int(chpos[1])]
            new_path[int(chpos[1])] = old_path[int(chpos[0])]
        else:
            d = ceil(random.rand(1,3)*N).tolist()[0];d.sort()
            a = int(d[0]);b=int(d[1]);c=int(d[2])
            if a !=b and b!=c:
                new_path = copy.deepcopy(old_path)
                new_path[a:c-1] = old_path[b-1:c-1]+old_path[a:b-1]
            else:
                new_path = self.changePath(old_path)
        return new_path

    #绘制路径
    def drawPath(Seq,dataMat,color='b'):
        m,n = shape(dataMat)
        px = (dataMat[Seq,0]).tolist()
        py = (dataMat[Seq,1]).tolist()
        px.append(px[0]);py.append(py[0])
        plt.plot(px,py,color)

    #绘制散点图
    def drawScatter(self,plt):
        px = (self.dataMat[:,0]).tolist()
        py = (self.dataMat[:,1]).tolist()
        plt.scatter(px,py,c='green',marker='o',s=60)
        i=65
        for x,y in zip(px,py):
            plt.annotate(str(chr(i)),xy=(x[0]+40,y[0]),color='black')
            i+=1

    #绘制趋势线
    def TrendLine(self,plt,color='b'):
        plt.plot(range(len(self.dist)),self.dist,color)

    def initBMNet(self,m,n,distMat):
        self.pathindx =list(range(m))
        randm.shuffle(self.pathindx);
        self.dist.append(self.pathLen(distMat,self.pathindx))
        return self.T0,self.pathindx,m

    #Boltzmann网络
    def train(self):
        [m,n] = shape(self.dataMat)
        distMat = self.distEclud(self.dataMat,self.dataMat.T)
        [T,curpath,MAX_M] = self.initBMNet(m,n,distMat)
        step = 0;
        while step <= self.MAX_ITER:
            m = 0;
            while m<= MAX_M:
                curdist = self.pathLen(distMat,curpath)
                newpath = self.changePath(curpath)
                newdist = self.pathLen(distMat,newpath)
                if (curdist > newdist):
                    curpath = newpath
                    self.pathindx.append(curpath)
                    self.dist.append(newdist)
                    self.iteration += 1
                else:
                    if random.rand() < self.boltzmann(newdist,curdist,T):
                        curpath = newpath
                        self.pathindx.append(curpath)
                        self.dist.append(newdist)
                        self.iteration += 1
            m += 1
            step += 1
            T = T*self.Lambda
        self.bestdist = min(self.dist)
        indxes = argmin(self.dist)
        self.bestpath = self.pathindx[indxes]
if __name__ == '__main__':
    bmNet = BoltzmannNet()
    bmNet.loadDataset("dataSet25.txt")
    bmNet.train()
    print('循环迭代',bmNet.iteration,'次')
    print('最优解',bmNet.bestdist)
    print('最佳路线',bmNet.bestpath)
    bmNet.drawScatter(plt)
    bmNet.drawPath(plt)
    plt.show()

    bmNet.TrendLine(plt)
    plt.show()



