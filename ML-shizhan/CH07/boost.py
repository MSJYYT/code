from numpy import *
from CH07 import adaboost
# 单层决策树生成函数
# lt=less than
# 分类器的构建(单纯地将某一特征上的所有取值与输入的阈值进行比较，
# 若制定lt为负，则特征值小于阈值的样本被标记为-1)
# 相反而知，若指定gt为负，则特征值大于阈值的样本被标记为-1
# dataMatrix - 数据矩阵
# dimen - 第dimen列，也就是第几个特征
# threshVal - 阈值
# threshIneq - 标志
# retArray - 分类结果
def stumpClassify(dataMatrix,dimen,threshVal,threshIneq):
    retArray = ones((shape(dataMatrix)[0],1))           #初始化retArray为1
    if threshIneq == 'lt':
        retArray[dataMatrix[:,dimen] <= threshVal] = -1 #如果小于阈值,则赋值为-1
    else:
        retArray[dataMatrix[:,dimen] > threshVal] = -1  #如果大于阈值,则赋值为-1
    return retArray

# stumpClassify分类器的预测值收到了特征、阈值和阈值两边到底哪边为正标签哪边为父标签的影响
# 所以有三重循环
# 第一重：遍历每个特征
# 第二重：对每个特征上依次设定不同的阈值
# 第三重：每个特征的每个阈值设定以后 还要依次以小、大于阈值作为依据调用分类器。得出预测结果。
# 将结果与真实结果对比，得出错误向量
# 通过错误向量得出加权错误值之后与当前的最小错误值进行对比，迭代后得到最终的最小错误
# Parameters:
#         dataArr - 数据矩阵
#         classLabels - 数据标签
#         D - 样本权重
# Returns:
#         bestStump - 最佳单层决策树信息
#         minError - 最小误差
#         bestClasEst - 最佳的分类结果
def buildStump(dataArr,classLabels,D):
    dataMatrix = mat(dataArr);labelMat = mat(classLabels).T
    m,n = shape(dataMatrix)
    numSteps = 10;bestStump = {};bestClasEst = mat(zeros((m,1)))
    minError = inf                                  #最小误差初始化为正无穷大
    for i in range(n):
        rangeMin = dataMatrix[:,i].min();rangeMax = dataMatrix[:,i].max();#找到特征中最小的值和最大值
        stepSize = (rangeMax-rangeMin)/numSteps      #计算步长
        for j in range(-1,int(numSteps)+1):         # 该维度上分隔线能取的范围内移动
            for inequal in ['lt','gt']:             #大于和小于的情况，均遍历。lt:less than，gt:greater than
                threshVal = (rangeMin + float(j) * stepSize)        #计算阈值
                predictedVals = stumpClassify(dataMatrix,i,threshVal,inequal)#计算分类结果

                errArr = mat(ones((m,1)))           #初始化误差矩阵
                errArr[predictedVals == labelMat] = 0       #分类正确的,赋值为0
                weightedError = D.T*errArr               #计算误差
                # print("split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f" % (
                # i, threshVal, inequal, weightedError))#找到误差最小的分类方式
                if weightedError < minError:
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump,minError,bestClasEst
if __name__ == '__main__':
    dataArr,classLabels = adaboost.loadSimpData()
    D = mat(ones((5,1))/5)
    bestStump,minError,bestClasEst = buildStump(dataArr,classLabels,D)
    print('bestStump:\n',bestStump)
    print('minError:\n',minError)
    print('bestClasEst\n',bestClasEst)