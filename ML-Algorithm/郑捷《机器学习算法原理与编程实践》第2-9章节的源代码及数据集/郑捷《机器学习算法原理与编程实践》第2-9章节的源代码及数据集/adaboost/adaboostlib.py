from numpy import *
import matplotlib.pyplot as plt
import sys

def LongToInt(value):
    assert isinstance(value, (int, long))
    return int(value & sys.maxint)

def loadDataSet(fileName) :
	recordlist = []
	fp = open(fileName,"rb")
	content = fp.read()
	fp.close()
	rowlist = content.splitlines()
	#print rowlist
	recordlist = [map(eval,row.split("\t")) for row in rowlist if row.strip()]
	#recordlist = [row.split('\t') for row in rowlist if row.strip()]
	m,n = shape(recordlist)
	dataSet = mat(recordlist)[:,:-1]
	for i in xrange(m) :
		if recordlist[i][-1] == 0.0 :
			recordlist[i][-1] = -1.0
	labels = mat(recordlist)[:,-1].T
	return dataSet,labels
	
def plotROC(predStrengths,classLabels) :
#	m,n = shape(classLabels)
#	print m,n
#	print type(classLabels)
#	print classLabels
	cur = (1.0,1.0)
	ySum = 0.0
	numPosClas = sum(array(classLabels) == 1.0)
	yStep = 1/float(numPosClas)
	xStep =1/float(len(classLabels) - numPosClas)
	sortedIndicies = predStrengths.argsort()
#	print sortedIndicies
	fig = plt.figure()
	fig.clf()
	ax = plt.subplot(111)
	for index in sortedIndicies.tolist()[0] :
		if classLabels[0,index] == 1.0 :
			delX = 0
			delY = yStep
		else :
			delX = xStep
			delY = 0
			ySum += cur[1]
		ax.plot([cur[0],cur[0] - delX],[cur[1],cur[1]-delY],c='b')
		cur = (cur[0] - delX,cur[1]-delY)
	ax.plot([0,1],[0,1],'b--')
	plt.xlabel('False positive rate')
	plt.ylabel('True positive rate')
	plt.title('ROC curve for AdaBoost horse colic datection system')
	ax.axis([0,1,0,1])
	plt.show()
	print "the Area Under the Cure is:",ySum*xStep

def decisionTree(dataSet,labellist,D) :
	dataMat = mat(dataSet)
	labelMat = mat(labellist).T
	m,n = shape(dataMat)
	numSteps = 10.0
	bestFeat = {}
	bestClass = mat(zeros((m,1)))
	minError = inf
	for i in xrange(n) :
		#rangeMin = dataMat[:,i].min()
#		print dataMat[:,i]
		rangeMin = dataMat[:,i].min()
		rangeMax = dataMat[:,i].max()
		stepSize = (rangeMax - rangeMin)/numSteps
		for j in xrange(-1,int(numSteps) + 1) :
			for operator in ['lt','gt'] :
				threshVal = (rangeMin + float(j) * stepSize)
				predictedVals = splitDataSet(dataMat,i,threshVal,operator)
#				print predictedVals
#				ddd
				errSet = mat(ones((m,1)))
				errSet[predictedVals == labelMat] = 0
				weightedError = D.T*errSet
				if weightedError < minError :
					minError = weightedError
					bestClass = predictedVals.copy()
					bestFeat['dim'] = i
					bestFeat['thresh'] = threshVal
					bestFeat['oper'] = operator
	return bestFeat,minError,bestClass
	
def adaBoostTrain(dataSet,labellist,numIt=40) :
	weakClassSet = []
	m = shape(dataSet)[0]
	D = mat(ones((m,1))/m)
#	print "D = ",D
	aggClassSet = mat(zeros((m,1)))
	for i in xrange(numIt) :
		bestFeat,error,EstClass = decisionTree(dataSet,labellist,D)
		alpha = float(0.5 * log((1-error)/max(error,1e-16)))
		bestFeat['alpha'] = alpha
		weakClassSet.append(bestFeat)
		wtx = multiply(-1*alpha*mat(labellist).T,EstClass)
		D = multiply(D,exp(wtx))
		D = D/D.sum()
		aggClassSet += alpha * EstClass
		totalErr = multiply(sign(aggClassSet) != mat(labellist).T,ones((m,1)))
		errorRate = totalErr.sum()/m
		print totalErr.sum(),m
		print "total error : ",errorRate
		if errorRate == 0.0 :break
	return weakClassSet,aggClassSet
	
def splitDataSet(dataMat,column,threshVal,operator) :
	retArray = ones((shape(dataMat)[0],1))
	if operator == 'lt' :
		retArray[dataMat[:,column] <= threshVal] = -1.0
	else :
		retArray[dataMat[:,column] > threshVal] = -1.0
	return retArray
	
def adaClassify(dataToClass,classifierArr) :
	dataMatrix = mat(dataToClass)
	m = shape(dataMatrix)[0]
	aggClassEst = mat(zeros((m,1)))
	for i in range(len(classifierArr)) :
		classEst = splitDataSet(dataMatrix,classifierArr[i]['dim'],
		classifierArr[i]['thresh'],classifierArr[i]['oper'])
		aggClassEst += classifierArr[i]['alpha']*classEst
	return sign(aggClassEst)
	
