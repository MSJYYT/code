from numpy import *
Data= [ [1,1,1,0,0],
        [2,2,2,0,0],
        [1,1,1,0,0],
        [5,5,5,0,0],
        [1,1,0,2,2],
        [0,0,0,3,3],
        [0,0,0,1,1]]
U,Sigma,VT=linalg.svd(Data)
#print(U)
#print(Sigma)
#print(VT)
#取前两个奇异值构成对角阵
Sigma2=mat([[Sigma[0],0],
            [0,Sigma[1]]])
A=U[:,:2]*Sigma2*VT[:2,:]
print(A.astype(int))
