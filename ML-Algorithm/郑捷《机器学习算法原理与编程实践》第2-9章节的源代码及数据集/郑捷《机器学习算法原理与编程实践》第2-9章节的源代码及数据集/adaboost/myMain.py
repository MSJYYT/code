from numpy import *
import sys
from adaboostlib import *
import matplotlib.pyplot as plt

dataArr,labelArr = loadDataSet('horseColicTraining.txt')
weakClassArr,aggClassEst = adaBoostTrain(dataArr,labelArr,numIt=10)
#print "weakClassArr:",weakClassArr
#print "aggClassEst:",aggClassEst
plotROC(aggClassEst.T,labelArr)

testArr,testLabelArr = loadDataSet('horseColicTest.txt') 
m,n = shape(testArr)
print m,n
ClassEst100 = adaClassify(testArr,weakClassArr)
errArr = mat(ones((67,1)))
totalError = errArr[ClassEst100 != mat(testLabelArr).T].sum()
print "totalError: ",totalError