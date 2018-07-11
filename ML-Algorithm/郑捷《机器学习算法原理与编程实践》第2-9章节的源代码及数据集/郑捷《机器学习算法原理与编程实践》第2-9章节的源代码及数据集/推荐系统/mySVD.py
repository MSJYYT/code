from numpy import *
import numpy as np
import operator

def cosSim(vecA, vecB) :
	return dot(vecA, vecB) / ((linalg.norm(vecA) * linalg.norm(vecB)) + eps)

dataSet = mat([[0,0,0,0,0,4,0,0,0,0,5],[0,0,0,3,0,4,0,0,0,0,3],
				[0,0,0,0,4,0,0,1,0,4,0],[3,3,4,0,0,0,0,2,2,0,0],
				[5,4,5,0,0,0,0,5,5,0,0],[0,0,0,0,5,0,1,0,0,5,0],
				[4,3,4,0,0,0,0,5,5,0,1],[0,0,0,4,0,4,0,0,0,0,4],
				[0,0,0,2,0,2,5,0,0,1,2],[0,0,0,0,5,0,0,0,0,4,0]])
testVect = mat([[1,0,0,0,0,0,0,1,2,0,0]])
eps = 1.0e-6
r = 2
rank = 2
m, n = shape(dataSet)
limit = min(m, n)
if r>limit :
	r = limit
U, S, VT = linalg.svd(dataSet.T)
print(U)
V = VT.T
Ur = U[: , :r]
print('\n')
print(Ur)
Sr = diag(S)[:r :, :r]
Vr = V[:, :r]
testresult = testVect * Ur * linalg.inv(Sr)
print('\n')
print(testresult)
print(Sr)
'''
resultarray = array([cosSim(testresult, vi) for vi in Vr])
descindx = argsort(-resultarray)[:rank]
print(descindx)
print(resultarray[descindx])
'''

