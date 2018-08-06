import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor,BaggingRegressor
from nltk.stem.snowball import SnowballStemmer
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

class searchrelevance(object):
    def __init__(self):
        self.df_train = []
        self.df_test = []
        self.desc = []
        self.df_all = []


    def loaddata(self):
        self.df_train = pd.read_csv('./input/train.csv',encoding='ISO-8859-1')
        self.df_test = pd.read_csv('./input/test.csv',encoding='ISO-8859-1')
        self.df_desc = pd.read_csv('./input/product_descriptions.csv')

        self.df_all = pd.concat((self.df_train,self.df_test),axis=0,ignore_index=True)

        self.df_all = pd.merge(self.df_all,self.df_desc,how='left',on='product_uid')
    def cleandata(self):
        stemmer = SnowballStemmer('english')

        def str_stemmer(s):
            return " ".join([stemmer.stem(word) for word in s.lower().split()])
        def str_common_word(str1,str2):
            return sum(int(str2.find(word)>=0) for word in str1.split())

        self.df_all['search_term'] = self.df_all['search_term'].map(lambda  x:str_stemmer(x))
        self.df_all['product_title'] = self.df_all['product_title'].map(lambda x:str_stemmer(x))
        self.df_all['product_description'] = self.df_all['product_description'].map(lambda x:str_stemmer(x))

        self.df_all['len_of_query'] = self.df_all['search_term'].map(lambda x:len(x.split())).astype(np.int64)
        self.df_all['commons_in_title'] = self.df_all.apply(lambda x:str_common_word(x['search_term'],x['product_title']),axis=1)
        self.df_all['commons_in_desc'] = self.df_all.apply(lambda x:str_common_word(x['search_term'],x['product_description']),axis=1)

        self.df_all = self.df_all.drop(['search_term','product_title','product_description'],axis=1)

        self.df_train = self.df_all.loc[self.df_train.index]
        self.df_test = self.df_all.loc[self.df_test.index]

        test_ids = self.df_test['id']

        y_train = self.df_train['relevance'].values

        X_train = self.df_train.drop(['id','relevance'],axis=1).values
        X_test = self.df_test.drop(['id','relevance'],axis=1).values

        pd.DataFrame(X_train).to_csv('./output/X_train.csv')
        pd.DataFrame(X_test).to_csv('./output/X_test.csv')
        pd.DataFrame(y_train).to_csv('./output/y_train.csv')
        pd.DataFrame(test_ids).to_csv('./output/test_ids.csv')
    def train(self):
        X_train = pd.read_csv('./output/X_train.csv',header=0,index_col=0)
        y_train = pd.read_csv('./output/y_train.csv',header=0,index_col=0)

        params = [1,2,4,5,6,7,8,9]
        test_scores = []
        for param in params:
            clf = RandomForestRegressor(n_estimators=30,max_depth=param)
            test_score = np.sqrt(-cross_val_score(clf,X_train,y_train.values.ravel(),cv=5,scoring='neg_mean_squared_error'))
            test_scores.append(np.mean(test_score))

        plt.plot(params,test_scores)
        plt.title('RandomForestResgressor CV error')
        plt.show()

    def predict(self):
        rf = RandomForestRegressor(n_estimators=30,max_depth=6)
        X_train = pd.read_csv('./output/X_train.csv', header=0, index_col=0)
        y_train = pd.read_csv('./output/y_train.csv', header=0, index_col=0)
        X_test = pd.read_csv('./output/X_test.csv',header=0,index_col=0)
        test_ids = pd.read_csv('./output/test_ids.csv',header=0,index_col=0)

        rf.fit(X_train,y_train.values.ravel())
        y_pred = rf.predict(X_test)

        pd.DataFrame({"id":test_ids.values.ravel(),"relevance":y_pred}).to_csv('./output/submission.csv',index=False)

if __name__ == '__main__':
    search = searchrelevance()
    # search.loaddata()
    # search.cleandata()
    # search.train()
    search.predict()
