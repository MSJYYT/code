import tensorflow as tf
b = tf.Variable(tf.zeros([100]))  #生成100维的向量，初始化为0
#生成784×100的随机矩阵w
w = tf.Variable(tf.random_uniform([784,100],-1,-1))
#输入placeholder
x = tf.placeholder(name='x')
#relu(wx+b)
relu = tf.nn.relu(tf.matmul(w,x)+b)
#根据relu函数的结果计算cost
C = [...]
s = tf.Session()
for step in range(0,10):
    #为输入创建一个100维的向量
    input =C #construct 100-D input array...
    #获取Cost，供给输入x
    result = s.run(C,feed_dict={x:input})
    print(step,result)
