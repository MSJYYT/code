from numpy import *

def loadDataSet(fileName):
    dataMat = [];labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]),float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat,labelMat

def selectJrand(i,m):
    # i是第一个alpha的下标，m都是alpha的总个数
    j = i
    while(j == i):
        # j是第二个alpha的下标
        # 如果j与i相同，则重新选取一个
        j = int(random.uniform(0,m))
    return j

def clipAlpha(aj,H,L):
    # 设置上下界限
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj

def smoSimple(dataMatIn,classLabels,C,toler,maxIter):
    # 函数输入：数据、标签集、常数、容错率、最大循环次数
    dataMatrix = mat(dataMatIn);labelMat = mat(classLabels).transpose()
    # m表示样本个数，n表示特征维度
    b = 0;m,n = shape(dataMatrix)
    alphas = mat(zeros((m,1)))
    iter = 0
    while(iter < maxIter):
        alphaPairsChanged = 0
        for i in range(m):
            # multiply是numpy的乘法函数，.T是转置
            # 第i个样本对应的判别结果
            # (dataMatrix*dataMatrix[i,:].T)是一个核函数计算
            fXi = float(multiply(alphas,labelMat).T *
                                 (dataMatrix*dataMatrix[i,:].T)) + b
            Ei = fXi - float(labelMat[i])
            # 误差，预测结果与真实类别值
            if ((labelMat[i]*Ei < toler) and (alphas[i] < C)) or\
                ((labelMat[i]*Ei > toler) and (alphas[i] > 0)):

                # 如果误差大于容错率或者alpha值不符合约束，则进入优化
                # 随机选择第二个alpha
                j = selectJrand(i,m)
                fXj = float(multiply(alphas,labelMat).T *
                                    (dataMatrix*dataMatrix[j,:].T)) + b
                # 计算第二个alpha的值
                Ej = fXj - float(labelMat[j])
                # 得到两个样本对应的两个alpha对应的误差值
                alphaIold = alphas[i].copy();
                alphaJold = alphas[j].copy();
                # 存储原本的alpha值
                if(labelMat[i] != labelMat[j]):
                   L = max(0,alphas[j] - alphas[i])
                   H = min(C,C + alphas[j] - alphas[i])
                else:
                   L = max(0,alphas[j] + alphas[i] - C)
                   H = min(C,alphas[j] + alphas[i])
                if L==H:
                    print('L = H')
                    continue
               # 计算上下阈值
               # 针对y1,y2的值相同与否，上下值也不同
                eta = 2.0 * dataMatrix[i,:]*dataMatrix[j,:].T - \
                      dataMatrix[i,:]*dataMatrix[i,:].T - \
                      dataMatrix[j,:]*dataMatrix[j,:].T
                if eta >= 0:
                   print('eta>=0')
                   continue
                alphas[j] -= labelMat[j]*(Ei-Ej)/eta
                # 由一个alpha确定另一个alpha
                b1 = b - Ei - labelMat[i]*(alphas[i] - alphaIold)*dataMatrix[i, :] * \
                dataMatrix[i, :].T - labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[i,:]*dataMatrix[j,:].T
                b2 = b - Ei - labelMat[i]*(alphas[i] - alphaIold)*dataMatrix[i, :]*\
                dataMatrix[j, :].T - labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[j,:]*dataMatrix[j,:].T
               # 更新两个b值
                if (0 < alphas[i]) and (C > alphas[i]):
                    b = b1
                elif (0 < alphas[j]) and (C > alphas[j]):
                    b = b2
                else:
                    b = (b1 + b2) / 2.0
                alphaPairsChanged += 1
                print('iter:%d i:%d,pairs changed %d' % (iter, i, alphaPairsChanged))

        if (alphaPairsChanged == 0):
            iter += 1
        else:
            iter = 0
        print('iteration number : %d' % iter)
        return b, alphas