import matplotlib.pyplot as plt

def g(x): 
	return 4.0*x**3
def f(x):
	return x**4

def armijo(x,d,a):
	c1 = 0.3
	now = f(x)
	next = f(x - a*d)

	count =30
	while next < now:
		a *= 2
		next = f(x - a*d)
		count-=1
		if count == 0:
			break
	count = 50
	while next > now-c1*a*d*d:
		a /=2
		next = f(x - a*d)
		count -=1
		if count == 0:
			break
	return a
def getA_quad(x,d,a):
	c1 = 0.3
	now = f(x)
	next = f(x - a*d)

	count =30
	while next < now:
		a *= 2
		next = f(x - a*d)
		count-=1
		if count == 0:
			break
	count = 50
	while next > now-c1*a*d*d:
		b=d*a*a/(now+d*a-next)
		b /=2
		if b<0:
			a /=2
		else:
			a = b
		next = f(x - a*d)
		count -=1
		if count == 0:
			break
	return a
if __name__ == '__main__':
	x =1.5
	a = 0.01
	# 固定学习率
	for i in range(1000):
		d = g(x)
		x -= d * a
		if i == 200:
			print(x)
		plt.scatter(i,x)
	print(x)
	plt.show()
	#回溯线性搜索
	for i in range(1000):
		d = g(x)
		a1 = armijo(x,d,a)
		x -= d * a1
		if i == 12:
			print(x,a1)
		plt.scatter(i,x)
	print(x,a1)
	plt.show()
	#插值法
	for i in range(1000):
		d = g(x)
		a1 = getA_quad(x, d, a)
		x -= d * a1
		if i == 12:
			print(x)
		plt.scatter(i, x)
	print(x)
	plt.show()

