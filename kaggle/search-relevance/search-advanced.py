import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor,BaggingRegressor
from nltk.stem.snowball import SnowballStemmer
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import Levenshtein
from gensim.utils import tokenize
from gensim.corpora.dictionary import Dictionary
from gensim.models.tfidfmodel import TfidfModel
from gensim.similarities import MatrixSimilarity
import nltk
from nltk.tokenize import word_tokenize
from gensim.models.word2vec import Word2Vec
from scipy import spatial
'''
id  - 表示（search_term，product_uid）对的唯一Id字段
product_uid - 产品的ID
product_title - 产品名称
product_description - 产品的文字说明（可能包含HTML内容）
search_term - 搜索查询
relevance - 给定id的相关性评级的平均值(最后的相关性得分)
'''
class searchrelevance(object):
    def __init__(self):
        self.df_train = []
        self.df_test = []
        self.desc = []
        self.df_all = []
        self.dictionary = []
    #读取数据集
    def loaddata(self):
        self.df_train = pd.read_csv('./input/train.csv',encoding='ISO-8859-1')
        self.df_test = pd.read_csv('./input/test.csv',encoding='ISO-8859-1')
        self.df_desc = pd.read_csv('./input/product_descriptions.csv')
        #训练集与测试集合并
        self.df_all = pd.concat((self.df_train,self.df_test),axis=0,ignore_index=True)
        #把产品介绍加入数据集
        self.df_all = pd.merge(self.df_all,self.df_desc,how='left',on='product_uid')
    #文本预处理
    def cleandata(self):
        stemmer = SnowballStemmer('english')
        #stem词干提取，有三种提取方法：SnowballStemmer、LancasterStemmer、PorterStemmer
        #具体效果可以找一些词试一试；以前用的词性还原lemmatization是把任何形式的词还原为一般形式
        #stem获得的词干不一定有实际意义
        def str_stemmer(s):
            return " ".join([stemmer.stem(word) for word in s.lower().split()])
        #计算str2中str1出现了多少次
        def str_common_word(str1,str2):
            return sum(int(str2.find(word)>=0) for word in str1.split())

        self.df_all['search_term'] = self.df_all['search_term'].map(lambda  x:str_stemmer(x))
        self.df_all['product_title'] = self.df_all['product_title'].map(lambda x:str_stemmer(x))
        self.df_all['product_description'] = self.df_all['product_description'].map(lambda x:str_stemmer(x))

        ###################################################
        #简单版：自制三个文本特征：关键词长度、标题中有多少关键词重合、描述中有多少关键词重合
        self.df_all['len_of_query'] = self.df_all['search_term'].map(lambda x:len(x.split())).astype(np.int64)
        self.df_all['commons_in_title'] = self.df_all.apply(lambda x:str_common_word(x['search_term'],x['product_title']),axis=1)
        self.df_all['commons_in_desc'] = self.df_all.apply(lambda x:str_common_word(x['search_term'],x['product_description']),axis=1)

        ############################################################
        #进阶版：自制文本特征
        #Levenshtein.ratio('hello','hello world')
        #计算几个字符串相似度：hamming（汉明距离）、distance（编辑距离）、ratio（莱温斯坦比）等
        self.df_all['dist_in_title'] = self.df_all.apply(lambda x:Levenshtein.ratio(x['search_term'],x['product_title']),axis=1)
        self.df_all['dist_in_desc'] = self.df_all.apply(lambda x:Levenshtein.ratio(x['search_term'],x['product_description']),axis=1)
        #TF-iDFTF-iDF稍微复杂点儿，因为它涉及到一个需要提前把所有文本计算统计一下的过程。
        #我们首先搞一个新的column，叫all_texts, 里面是所有的texts。
        # （我并没有算上search term, 因为他们不是一个结构完整的句子，可能会影响tfidf的学习）。为了防止句子格式不完整，我们也强制给他们加上句号。
        self.df_all['all_texts'] = self.df_all['product_title']+'.'+self.df_all['product_description']+'.'
        #然后，我们取出所有的单字，做成一个我们的单词字典：
        # （这里我们用gensim，为了更加细致的分解TFIDF的步骤动作；
        # 其实sklearn本身也有简单好用的tfidf模型，详情见第二课stock news基本版教程）
        #Tokenize可以用各家或者各种方法，就是把长长的string变成list of tokens。
        # 包括NLTK，SKLEARN都有自家的解决方案。或者你自己直接用str自带的split()方法，
        # 也是一种tokenize。只要记住，你这里用什么，那么之后你的文本处理都得用什么。
        self.dictionary = Dictionary(list(tokenize(x,errors='ignore')) for x in self.df_all['all_texts'].values)

    def finaldata(self):
        #####################################################
        #TFIDF增加两个特征
        #扫便我们所有的预料，并且转化成简单的单词的个数计算。
        def mycorpus():
            for x in self.df_all['all_texts'].values:
                yield self.dictionary.doc2bow(list(tokenize(x, errors='ignore')))

        corpus = mycorpus()
        #把已经变成BoW向量的数组，做一次TFIDF的计算。
        tfidf = TfidfModel(corpus)
        def totfidf(text):
            res = tfidf[self.dictionary.doc2bow(list(tokenize(text, errors='ignore')))]
            return res

        def cos_sim(text1, text2):
            tfidf1 = totfidf(text1)
            tfidf2 = totfidf(text2)
            index = MatrixSimilarity([tfidf1], num_features=len(self.dictionary))
            sim = index[tfidf2]
            # 本来sim输出是一个array，我们不需要一个array来表示，
            # 所以我们直接cast成一个float
            return float(sim[0])
        #例如：计算两个两个tfidf的cos
        # text1 = 'hello world'
        # text2 = 'hello from the other side'
        # cos_sim(text1, text2)
        self.df_all['tfidf_cos_sim_in_title'] = self.df_all.apply(lambda x: cos_sim(x['search_term'], x['product_title']), axis=1)
        self.df_all['tfidf_cos_sim_in_desc'] = self.df_all.apply(lambda x: cos_sim(x['search_term'], x['product_description']), axis=1)
        #至此，增加了两个feature

        ############################################
        #word2vec增加两个文本特征
        # nltk也是自带一个强大的句子分割器。
        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        # tokenizer.tokenize(self.df_all['all_texts'].values[0])
        #先把长文本搞成list of 句子
        sentences = [tokenizer.tokenize(x) for x in self.df_all['all_texts'].values]
        #其实这些sentences不需要这些层级关系，他们都是平级的，所以：
        #我们把list of lists 给 flatten了。
        sentences = [y for x in sentences for y in x]

        #再把句子变成list of 单词
        w2v_corpus = [word_tokenize(x) for x in sentences]

        #训练模型
        model = Word2Vec(w2v_corpus, size=128, window=5, min_count=5, workers=4)

        # 先拿到全部的vocabulary
        vocab = model.wv.vocab
        # 得到任意text的vector
        def get_vector(text):
            res = np.zeros([128])
            count = 0
            for word in word_tokenize(text):
                if word in vocab:
                    res += model[word]
                    count += 1
            return res / count

        def w2v_cos_sim(text1, text2):
            try:
                w2v1 = get_vector(text1)
                w2v2 = get_vector(text2)
                sim = 1 - spatial.distance.cosine(w2v1, w2v2)
                return float(sim)
            except:
                return float(0)

        #计算两个文本在Word2vec下的cos距离，作为两个特征
        self.df_all['w2v_cos_sim_in_title'] = self.df_all.apply(lambda x:w2v_cos_sim(x['search_term'],x['product_title']),axis=1)
        self.df_all['w2v_cos_sim_in_desc'] = self.df_all.apply(lambda x:w2v_cos_sim(x['search_term'],x['product_description']),axis=1)
        print(self.df_all.head())



        self.df_all = self.df_all.drop(['search_term','product_title','product_description','all_texts'],axis=1)

        self.df_train = self.df_all.loc[self.df_train.index]
        self.df_test = self.df_all.loc[self.df_test.index]
        test_ids = self.df_test['id']
        y_train = self.df_train['relevance'].values
        X_train = self.df_train.drop(['id','relevance'],axis=1).values
        X_test = self.df_test.drop(['id','relevance'],axis=1).values

        pd.DataFrame(X_train).to_csv('./output_advance/X_train.csv')
        pd.DataFrame(X_test).to_csv('./output_advance/X_test.csv')
        pd.DataFrame(y_train).to_csv('./output_advance/y_train.csv')
        pd.DataFrame(test_ids).to_csv('./output_advance/test_ids.csv')

    def train(self):
        X_train = pd.read_csv('./output_advance/X_train.csv',header=0,index_col=0)
        y_train = pd.read_csv('./output_advance/y_train.csv',header=0,index_col=0)

        params = [5,6,7,8,9,10,12]
        test_scores = []
        for param in params:
            clf = RandomForestRegressor(n_estimators=30,max_depth=param)
            test_score = np.sqrt(-cross_val_score(clf,X_train,y_train.values.ravel(),cv=5,scoring='neg_mean_squared_error'))
            test_scores.append(np.mean(test_score))

        plt.plot(params,test_scores)
        plt.title('RandomForestResgressor CV error')
        plt.show()

if __name__ == '__main__':
    advance = searchrelevance()
    # advance.loaddata()
    # advance.cleandata()
    #
    # advance.finaldata()
    advance.train()