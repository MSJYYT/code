import numpy as np
from numpy import *
import matplotlib.pyplot as plt
#from SVMClass import *

'''
a = mat(zeros((4,2)) +2)
print(a)
b = mat(ones((2,4)))
print(b)
#c = multiply(a,b)
c = a * b
print(c)
'''
a = mat(zeros((200, 1)))
a[1,0] = 2
a[3,0] = 4
a[5,0] = 6
a[7,0] = 8
b = nonzero((a.A > 0) * (a.A < 9))[0]
print(b)


'''
svm = PlattSVM()
svm.C = 100                                                                                   #惩罚因子
svm.tol = 0.001                                                                            #容错律
svm.maxIter = 10000
svm.kValue['Gaussian'] = 3.0                                                  #核函数
svm.loadDataSet('svm.txt')
svm.train()
print(shape(svm.sptVects)[0])
print("b: ", svm.b)
svm.scatterplot(plt)
plt.show()
'''

