Tn = 100.0
Sn = Tn/2

for i in range(1,11):
    Tn += 2*Sn
    Sn /= 2         #下次反弹减半
    #print(Sn)
print('total high is %f'%Tn)
print('第十次高%f'%Sn)