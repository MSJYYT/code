#将一个整数分解质因数，例如90=2*3*3*5



from sys import stdout
n = int(input('input number:\n'))
print('n=%d'%n)

for i in range(2,n+1):  #从2到n一个一个作为除数
    while n !=i:        #
        if n%i == 0:    #如果能除尽
            # stdout.write(str(i))  #就把这个除数输出
            # stdout.write('*')
            print(i,end='')
            print('*',end='')
            n = n//i        #在把整数部分作为n再开始找
        else:               #除不尽了就找到最后一个质数了
            break
print('%d'%n)