#一个数如果恰好等于它的因子之和，这个数就成为完数，例如6=1+2+3

# from sys import stdout
# from functools import reduce
# for j in range(2,1001):
#     k = []
#     n = -1          #用来计数几个因子
#     s= j            #把该数j作为被减数，如果依次被因子减为0则说明找到了
#     for i in range(1,j):
#         if j%i == 0: #只要除完没余数就是因子
#             n += 1
#             s -= i
#             k.append(i)
#     if s == 0:
#         print(j)
#         for i in range(n):
#             print(k[i],end='  ')
#             stdout.write('')
#         print(k[n])
#         sum = reduce(lambda x,y:x+y,k)
#         print('sum=%d' %sum)

from functools import reduce

for i in range(2,1001):
    Tn = []
    for j in range(1,i):
        if i % j == 0:
            Tn.append(j)
    Sn = reduce(lambda x,y:x+y,Tn)
    if Sn == i:
        print(i)
        print(Tn)