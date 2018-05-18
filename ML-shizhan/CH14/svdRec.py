from numpy import linalg as la
from numpy import *

def printMat(inMat,thresh =0.8):
    for i in range(32):
        for k in range(32):
            if float(inMat[i,k]) > thresh:
                print(1,end='')
            else:print(0,end='')
        print('')

def imgCompress(numSV=3,thresh=0.8):
    myl = []
    for line in open('0_5.txt').readlines():
        newRow = []
        for i in range(32):
            newRow.append(int(line[i]))
        myl.append(newRow)
    myMat = mat(myl)
    print('***original matrix***')
    printMat(myMat,thresh)
    U,Sigma,VT = la.svd(myMat)
    SigRecon = mat(zeros((numSV,numSV)))
    for k in range(numSV):
        SigRecon[k,k] = Sigma[k]
    reconMat = U[:,:numSV] * SigRecon*VT[:numSV,:]
    print('***reconstructed matrix using %d singular values***' % numSV)
    printMat(reconMat,thresh)

imgCompress(2)