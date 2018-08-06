import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from datetime import date
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from gensim.models.word2vec import Word2Vec
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt


data = pd.read_csv('./input/Combined_News_DJIA.csv')
# print(data.head())

train = data[data['Date'] < '2015-01-01']
test = data[data['Date'] > '2014-12-31']

X_train = train[train.columns[2:]]
corpus = X_train.values.flatten().astype(str)
X_train = X_train.values.astype(str)
X_train = np.array([' '.join(x) for x in X_train])

X_test = test[test.columns[2:]]
X_test = X_test.values.astype(str)
X_test = np.array([' '.join(x) for x in X_test])

y_train = train['Label'].values
y_test = test['Label'].values

corpus = [word_tokenize(x) for x in corpus]
X_train = [word_tokenize(x) for x in X_train]
X_test = [word_tokenize(x) for x in X_test]
#print(corpus[:3])

#停止词
stop = stopwords.words('english')
#数字
def hasNumbers(inputString):
    return bool(re.search(r'\d',inputString))
#特殊符号
def isSymbol(inputString):
    return bool(re.match(r'[^\w]',inputString))
#lemma还原单词
wordnet_lemmatizer = WordNetLemmatizer()

def check(word):
    #如果需要这个单词，则True,如果应该去除，则False
    word = word.lower()
    if word in stop:
        return False
    elif hasNumbers(word) or isSymbol(word):
        return False
    else:
        return True

#综合上述方法，删除无用单词
def preprocessing(sen):
    res = []
    for word in sen:
        if check(word):
            word = word.lower().replace("b'",'').replace('b"','').replace('"','').replace("'",'')
            res.append(wordnet_lemmatizer.lemmatize(word))
    return res

#处理这三组数据
corpus = [preprocessing(x) for x in corpus]
X_train = [preprocessing(x) for x in X_train]
X_test = [preprocessing(x) for x in X_test]

#训练NLP模型，简单的word2vec
'''
1、训练模型定义

from gensim.models import Word2Vec
model = Word2Vec(sentences, sg=1, size=100,  window=5,  min_count=5,  negative=3, sample=0.001, hs=1, workers=4)
参数解释：
    1.sg=1是skip-gram算法，对低频词敏感；默认sg=0为CBOW算法。CBOW是从原始语句推测目标字词；而Skip-Gram正好相反，是从目标字词推测出原始语句。CBOW对小型数据库比较合适，而Skip-Gram在大型语料中表现更好。 
    2.size是输出词向量的维数，值太小会导致词映射因为冲突而影响结果，值太大则会耗内存并使算法计算变慢，一般值取为100到200之间。
    3.window是句子中当前词与目标词之间的最大距离，3表示在目标词前看3-b个词，后面看b个词（b在0-3之间随机）。
    4.min_count是对词进行过滤，频率小于min-count的单词则会被忽视，默认值为5。
    5.negative和sample可根据训练结果进行微调，sample表示更高频率的词被随机下采样到所设置的阈值，默认值为1e-3。
    6.hs=1表示层级softmax将会被使用，默认hs=0且negative不为0，则负采样将会被选择使用。
    7.workers控制训练的并行，此参数只有在安装了Cpython后才有效，否则只能使用单核。

2、训练后的模型保存与加载
    model.save(fname)
    model = Word2Vec.load(fname)
    
3、模型使用（词语相似度计算等）
    model.most_similar(positive=['woman', 'king'], negative=['man'])
    #输出[('queen', 0.50882536), ...]
 
    model.doesnt_match("breakfast cereal dinner lunch".split())
    #输出'cereal'
 
    model.similarity('woman', 'man')
    #输出0.73723527
 
    model['computer']  # raw numpy vector of a word
    #输出array([-0.00449447, -0.00310097,  0.02421786, ...], dtype=float32)
'''
model = Word2Vec(corpus,size=128,window=5,min_count=5,workers=4)

#用NLP模型表达X_train
vocab = model.wv.vocab
def get_vector(word_list):
    res = np.zeros([128])
    count = 0
    for word in word_list:
        if word in vocab:
            res += model[word]
            count += 1
    return res/count
wordlist_train = X_train
wordlist_test = X_test

X_train = [get_vector(x) for x in X_train]
X_test = [get_vector(x) for x in X_test]

#建立训练模型
params = [0.1,0.5,1,3,5,7,10,12,16,20,25,30,35,40]
test_sores = []
for param in params:
    clf = SVR(gamma=param)
    test_sore = cross_val_score(clf,X_train,y_train,cv=3,scoring='roc_auc')
    test_sores.append(np.mean(test_sore))

plt.plot(params,test_sores)
plt.title('SVM AUC Score')
#plt.show()
def transform_to_matrix(x,padding_size=256,vec_size=128):
    res = []
    for sen in x:
        matrix = []
        for i in range(padding_size):
            try:
                matrix.append(model[sen[i]].tolist())
            except:
                # 这里有两种except情况，
                # 1. 这个单词找不到
                # 2. sen没那么长
                # 不管哪种情况，我们直接贴上全是0的vec
                matrix.append([0]*vec_size)
        res.append(matrix)
    return res
X_train = transform_to_matrix(wordlist_train)
X_test = transform_to_matrix(wordlist_test)

X_train = np.array(X_train)
X_test = np.array(X_test)


from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Convolution2D,MaxPooling2D
from keras.layers.core import Dense,Dropout,Activation,Flatten
from keras import backend


# X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1], X_train.shape[2])
# X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1], X_test.shape[2])


# 根据不同的backend定下不同的格式
#该选择结构可以灵活对theano和tensorflow两种backend生成对应格式的训练数据格式。
# 举例说明：'th'模式，即Theano模式会把100张RGB三通道的16×32（高为16宽为32）
# 彩色图表示为下面这种形式（100,3,16,32），Caffe采取的也是这种方式。
# 第0个维度是样本维，代表样本的数目，第1个维度是通道维，代表颜色通道数。
# 后面两个就是高和宽了。而TensorFlow，即'tf'模式的表达形式是（100,16,32,3），
# 即把通道维放在了最后。这两个表达方法本质上没有什么区别。
if backend.image_dim_ordering() == 'th':
    X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1], X_train.shape[2])
    X_test = X_test.reshape(X_test.shape[0], 1, X_train.shape[1], X_train.shape[2])
    input_shape = (1, X_train.shape[1], X_train.shape[2])
else:
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
    X_test = X_test.reshape(X_test.shape[0], X_train.shape[1], X_train.shape[2], 1)
    input_shape = (X_train.shape[1], X_train.shape[2], 1)

print(X_train.shape)
print(X_test.shape)

#set parameters:
batch_size = 32
n_filter = 16
filter_length = 4
nb_epoch = 5
n_pool = 2

#新建一个sequential的模型
model = Sequential()
#keras.layers.convolutional.Conv2D(filters, kernel_size,strides=(1,1),
#                                  padding='valid', data_format=None,
#                                  dilation_rate=(1,1), activation=None,
#                                   use_bias=True, kernel_initializer='glorot_uniform',
#                                  bias_initializer='zeros', kernel_regularizer=None,
#                                   bias_regularizer=None, activity_regularizer=None,
#                                   kernel_constraint=None, bias_constraint=None)

#二维卷积层对二维输入进行滑动窗卷积，当使用该层作为第一层时，应提供input_shape参数。
#filters：卷积核的数目；
#kernel_size：卷积核的尺寸；
#strides：卷积核移动的步长，分为行方向和列方向；
#padding：边界模式，有“valid”，“same”或“full”，full需要以theano为后端；
#卷积层1
model.add(Convolution2D(n_filter,filter_length,filter_length,input_shape=input_shape))
#激活层
model.add(Activation('relu'))

#卷积层2
model.add(Convolution2D(n_filter,filter_length,filter_length))
#激活层
#keras.layers.core.Activation(activation)
#激活层对一个层的输出施加激活函数。
#预定义激活函数：
#softmax，softplus，softsign，relu，tanh，sigmoid，hard_sigmoid，linear等。
model.add(Activation('relu'))

#池化层
#keras.layers.pooling.MaxPooling2D(pool_size=(2,2), strides=None, padding='valid',
#                                   data_format=None)
#对空域信号进行最大值池化。
#pool_size：池化核尺寸；
#strides：池化核移动步长；
#padding：边界模式，有“valid”，“same”或“full”，full需要以theano为后端；
model.add(MaxPooling2D(pool_size=(n_pool,n_pool)))
#神经元随机失活
#keras.layers.core.Dropout(p)
#为输入数据施加Dropout。Dropout将在训练过程中每次更新参数时随机断开
# 一定百分比（p）的输入神经元连接，Dropout层用于防止过拟合。
model.add(Dropout(0.25))
#拉成一维数据
#keras.layers.core.Flatten()
#Flatten层用来将输入“压平”，即把多维的输入一维化，
# 常用在从卷积层到全连接层的过渡。Flatten不影响batch的大小。
#例子：
    #model =Sequential()
    #model.add(Convolution2D(64,3, 3, border_mode='same', input_shape=(3, 32, 32)))
    ## now:model.output_shape == (None, 64, 32, 32)
    #model.add(Flatten())
    ## now:model.output_shape == (None, 65536)
model.add(Flatten())

#后面接一个ANN全连接层
#全连接层1
#keras.layers.core.Dense(units,activation=None, use_bias=True,
#                        kernel_initializer='glorot_uniform',bias_initializer='zeros',
#                        kernel_regularizer=None, bias_regularizer=None,
#                        activity_regularizer=None, kernel_constraint=None,
#                        bias_constraint=None)

#units：输出单元的数量，即全连接层神经元的数量，作为第一层的Dense层必须指定input_shape。
model.add(Dense(128))
#激活层
model.add(Activation('relu'))
#随机失活
model.add(Dropout(0.5))
#全连接层2
model.add(Dense(1))
#softmax评分
model.add(Activation('softmax'))

#编译模型
'''
compile(self,optimizer, loss, metrics=[], sample_weight_mode=None)
编译用来配置模型的学习过程，其参数有：
        optimizer：字符串（预定义优化器名）或优化器对象；
        loss：字符串（预定义损失函数名）或目标函数；
        metrics：列表，包含评估模型在训练和测试时的网络性能的指标，典型用法是metrics=['accuracy']；
'''

model.compile(loss='mse',optimizer='adadelta',metrics=['accuracy'])
#训练模型
'''
fit(self,x, y, batch_size=32, epochs=10, verbose=1, 
    callbacks=None,validation_split=0.0, validation_data=None, 
    shuffle=True, class_weight=None,sample_weight=None, initial_epoch=0)

        verbose：日志显示，0为不在标准输出流输出日志信息，
                    1为输出进度条记录，2为每个epoch输出一行记录；
        
        validation_split：0~1之间的浮点数，用来指定训练集的一定比例数据作为验证集。
                验证集将不参与训练，并在每个epoch结束后测试的模型的指标，如损失函数、精确度等；
        
        validation_data：形式为（X，y）的tuple，是指定的验证集。此参数将覆盖validation_spilt。
'''
model.fit(X_train,y_train,batch_size=batch_size,nb_epoch=nb_epoch,verbose=0)
#评估模型
score = model.evaluate(X_test,y_test,verbose=0)
print('Teset score:',score[0])
print('Teset accuracy:',score[1])

