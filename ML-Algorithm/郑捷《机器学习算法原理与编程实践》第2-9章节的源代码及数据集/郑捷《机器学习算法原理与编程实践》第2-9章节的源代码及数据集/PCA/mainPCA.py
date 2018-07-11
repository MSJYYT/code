from numpy import *
import sys,os
#from MyPCA import *
from MyPCA import *

reload (sys)
sys.setdefaultencoding('utf-8')

ef = Eigenfaces()
#ef.dist_metric = ef.distEclud
ef.loadimgs("att\\")
ef.compute()
testImg = ef.X[30]
#print ef.Y
print "shijizhi = ",ef.Y[30],">","yucezhi = ", ef.predict(testImg)
# E = []
# X = mat(zeros((10,10304)))
# for i in xrange(16) :
	# X = ef.Mat[i*10:(i+1)*10,:].copy()
	# X = X.mean(axis = 0)
	# imgs = X.reshape(112,92)
	# E.append(imgs)
# ef.subplot(title="AT&T Eign Facedatabase", images = E)
