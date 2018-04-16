# 一个整数，它加上100后是一个完全平方数，再加上268又是一个完全平方数
# 求该数

import math

for i in range(100000):
    #if (math.sqrt(i+100) - int(math.sqrt(i+100)) == 0):
        x = int(math.sqrt(i+100))
        #if (math.sqrt(i + 268) - int(math.sqrt(i + 268)) == 0):
        y = int(math.sqrt(i+268))
        if (x*x == (i+100)) and (y*y == (i+268)):
            print(i)