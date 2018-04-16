# 利润小于等于10万时，奖金提10%
# 10万<利润<=20万时，低于10万的按10%提成，高于10万的按7.5%
# 20万<利润<=40万时，高于20万按5%
# 40万<利润<=60万时，高于40万的按3%
# 60万<利润<=100万时，高于60万按1.5%
# 100万<利润时，超过100万的按1%
# 求奖金

bonus1 = 100000*0.1
bonus2 = bonus1 + 100000*0.075
bonus4 = bonus2 + 200000*0.05
bonus6 = bonus4 + 200000*0.03
bonus10 = bonus6 + 400000*0.015

i  = int(input('input gain:\n'))
if i <= 100000:
    bonus = i * 0.1
elif i <= 200000:
    bonus = bonus1 + (i-100000)*0.075
elif i <= 400000:
    bonus = bonus2 + (i - 200000)*0.05
elif i <= 600000:
    bonus = bonus4 + (i-400000)*0.03
elif i <= 1000000:
    bonus = bonus6 + (i-600000)*0.015
else:
    bonus = bonus10 + (i-1000000)*0.01

print('bonus=',bonus)