from numpy import *

    #创建实验样本，真实样本可能差很多，需要对真实样本做一些处理，比如
    #去停用词(stopwords)，词干化(stemming)等等，处理完后得到更"clear"的数据集，方便后续处理

def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]    #文档类别，1代表存在侮辱性的文字，0代表不存在
    return postingList,classVec

def createVocabList(dataSet):
    # 将所有文档所有（去重的）词都存到一个列表中，可用set()函数去重。
    # 用上set()函数操作符号|，取并集，或者写两重循环用vocabSet.add()
    # return list(set([word for doc in dataSet for word in doc])
    # [word for doc in dataSet for word in doc]: 用列表推导式将dataSet转为1维列表，
    # set(XXX)： 将这个列表去重转为集合
    # list(set(XXX)): 又转回来
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)#创建两个集合的并集，去除重复元素
    return list(vocabSet)

def setOfWords2Vec(vocabList, inputSet): #输入参数为词汇表及某个文档
    returnVec = [0]*len(vocabList) #文档向量和词汇表长度一样
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:print("the word : %s is not in my Vocabulary!" % word)
    return returnVec#输出文档向量，向量的每一元素为1或0，分别表示词汇表中的单词在输入文档中是否出现

# listOPosts,listClasses = loadDataSet()
# myVocabList = createVocabList(listOPosts)
#
# print(myVocabList) #不会出现重复的词
# print(listOPosts[5])
# Vec = setOfWords2Vec(myVocabList,listOPosts[5])
# print(Vec)

def trainNB0(trainMatrix,trainCategory):#输入参数为文档向量矩阵trainMatrix，文档类别所构成的向量trainCategory
    #trainMatrix组成，每行列数都是所有不重复的单词数，每行都是setOfWords2Vec转化后的文档向量
    #计算文档的数目，行数
    numTrainDocs = len(trainMatrix)
    #计算单词的数目，每行个数，既不重复的单词个数
    numWords = len(trainMatrix[0])
    # 计算类别的概率，abusive为1，not abusive为0
    pAbusive = sum(trainCategory)/float(numTrainDocs)
    # 初始化计数器，1行numWords列，p0是not abusive，p1是abusive
    # p0Num = zeros(numWords);p1Num = zeros(numWords)
    p0Num = ones(numWords);p1Num = ones(numWords)

    # 初始化分母
    # p0Denom = 0.0;p1Denom = 0.0
    p0Denom = 2.0;p1Denom = 2.0
    #遍历文档
    for i in range(numTrainDocs):
        # 计算abusive对应的词汇的数目，trainMatrix为0-1值形成的向量
        if trainCategory[i] == 1:
            # p1Num存储的是每个词出现的次数
            p1Num += trainMatrix[i]

            # p1Denom存储的是词的总数目
            p1Denom += sum(trainMatrix[i])
            # 计算not abusive词汇的数目
        else:
            # 每个词在not abusive下出现的次数
            p0Num += trainMatrix[i]
            # not abusive下的总词数
            p0Denom += sum(trainMatrix[i])
    # 计算abusive下每个词出现的概率
    #p1Vect = p1Num/p1Denom
    p1Vect = log(p1Num / p1Denom)
    # 计算not abusive下每个词出现的概率
    p0Vect = log(p0Num/p0Denom)
    # 返回词出现的概率和文档为abusive的概率，not abusive的概率为1-pAbusive
    return p0Vect,p1Vect,pAbusive

# listOPosts,listClasses = loadDataSet()
# myVocabList = createVocabList(listOPosts)
# trainMat = []
# for postinDoc in listOPosts:
#     trainMat.append(setOfWords2Vec(myVocabList,postinDoc))
#
# p0V,p1V,pAb = trainNB0(trainMat,listClasses)
def classifyNB(vec2Classify,p0Vec,p1Vec,pClass1):
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + log(1-pClass1)
    if p1 > p0:
        return 1
    else:
        return 0

def testingNB():
    listOPosts,listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList,postinDoc))
    p0V,p1V,pAb = trainNB0(array(trainMat),array(listClasses))

    testEntry = ['love','my','dalmation']
    thisDoc = array(setOfWords2Vec(myVocabList,testEntry))
    print(testEntry,'classified as:',classifyNB(thisDoc,p0V,p1V,pAb))

    testEntry = ['stupid','garbage']
    thisDoc = array(setOfWords2Vec(myVocabList,testEntry))
    print(testEntry,'classfied as;',classifyNB(thisDoc,p0V,p1V,pAb))

if __name__=="__main__":
    testingNB()