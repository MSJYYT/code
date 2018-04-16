from CH02 import kNN
import matplotlib
import matplotlib.pyplot as plt
from numpy import *
from os import listdir

def testplot():
    # group,labels = kNN.createDataSet()
    # print(kNN.classify0([0,0],group,labels,3))
    datingDataMat, datingLabels = kNN.file2matrix('datingTestSet.txt')
    # print(datingLabels[0:20])

    # add_subplot(mnp)添加子轴、图。subplot（m,n,p）或者subplot（mnp）
    # 此函数最常用：subplot是将多个图画到一个平面上的工具。
    # 其中，m表示是图排成m行，n表示图排成n列，也就是整个figure中有n个图是排成一行的，
    # 一共m行，如果第一个数字是2就是表示2行图。p是指你现在要把曲线画到figure中哪个图上，
    # 最后一个如果是1表示是从左到右第一个位置。
    # ax.scatter(datingDataMat[:,1],datingDataMat[:,2],
    # 15.0*array(datingLabels),15.0*array(datingLabels))
    # 以第二列和第三列为x,y轴画出散列点，给予不同的颜色和大小
    # scatter（x,y,s=1,c="g",marker="s",linewidths=0）
    # s:散列点的大小,c:散列点的颜色，marker：形状，linewidths：边框宽度
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.figure(figsize=(8, 5), dpi=80)
    axes = plt.subplot(111)
    # 将三类数据分别取出来
    # x轴代表飞行的里程数
    # y轴代表玩视频游戏的百分比
    type1_x = []
    type1_y = []
    type2_x = []
    type2_y = []
    type3_x = []
    type3_y = []

    for i in range(len(datingLabels)):
        if datingLabels[i] == 1:  # 不喜欢
            type1_x.append(datingDataMat[i][0])
            type1_y.append(datingDataMat[i][1])

        if datingLabels[i] == 2:  # 魅力一般
            type2_x.append(datingDataMat[i][0])
            type2_y.append(datingDataMat[i][1])

        if datingLabels[i] == 3:  # 极具魅力

            type3_x.append(datingDataMat[i][0])
            type3_y.append(datingDataMat[i][1])

    type1 = axes.scatter(type1_x, type1_y, s=20, c='red')
    type2 = axes.scatter(type2_x, type2_y, s=40, c='green')
    type3 = axes.scatter(type3_x, type3_y, s=50, c='blue')
    # plt.scatter(matrix[:, 0], matrix[:, 1], s=20 * numpy.array(labels),
    #             c=50 * numpy.array(labels), marker='o',
    #             label='test')
    plt.xlabel(u'每年获取的飞行里程数')
    plt.ylabel(u'玩视频游戏所消耗的事件百分比')
    axes.legend((type1, type2, type3), (u'不喜欢', u'魅力一般', u'极具魅力'), loc=2)
    plt.show()

def datingClassTest():
    #将数据集中10%作为测试，其余90%作为训练
    hoRatio = 0.1
    datingDataMat,datingLabels = kNN.file2matrix('datingTestSet.txt')
    normMat,ranges,minVals = kNN.autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = kNN.classify0(normMat[i,:],normMat[numTestVecs:m,:],     #normMat[i,:]归一化后的第i行测试数据；normMat[numTestVecs:m,除去numTestVecs行测试数据后的样本数据
                                     datingLabels[numTestVecs:m],4)
        print("the classifier came back with: %d, the real answer is: %d, result is :%s" %
              (classifierResult, datingLabels[i], classifierResult == datingLabels[i]))
        if (classifierResult != datingLabels[i]): errorCount += 1.0
    print("the total error rate is: %f" % (errorCount / float(numTestVecs)))
    print(errorCount)

def classfyPerson():
    resultList = ["not at all","in small doses","in large doses"]
    percentTats = float(input("percentage of time spent playing video games?"))
    ffMiles = float(input("frequent flier miles earned per year?"))
    iceCream = float(input("liters of ice cream consumed per year?"))
    datingDataMat, datingLabels = kNN.file2matrix('datingTestSet.txt')
    normMat, ranges, minVals = kNN.autoNorm(datingDataMat)
    inArr = array([ffMiles,percentTats,iceCream])
    classfierResult = kNN.classify0((inArr-minVals)/ranges,normMat,datingLabels,3)
    print("you will probably like this person:",resultList[classfierResult - 1])

def handwritingClassTest():
    hwLabels = []
    #加载训练数据
    trainingFileList = listdir('trainingDigits')
    m = len(trainingFileList)#m个样本
    trainingMat = zeros((m,1024))
    for i in range(m):
        #从文件名中解析出当前图像的标签，也就是数字是几
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:] = kNN.img2vector('trainingDigits/%s' % fileNameStr)
    #加载测试数据
    testFileList = listdir('testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = kNN.img2vector('testDigits/%s' % fileNameStr)

        classifierResult = kNN.classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print("the classifier came back with: %d, the real answer is: %d, The predict result is: %s" %
              (classifierResult, classNumStr, classifierResult == classNumStr))
        if (classifierResult != classNumStr): errorCount += 1.0
    print("\nthe total number of errors is: %d / %d" % (errorCount, mTest))
    print("\nthe total error rate is: %f" % (errorCount / float(mTest)))


if __name__=="__main__":
    #datingClassTest()
    #classfyPerson()
    #testplot()
    handwritingClassTest()