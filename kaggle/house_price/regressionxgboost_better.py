import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import AdaBoostRegressor
from xgboost import XGBRegressor
class Houseprice(object):
    #构造方法
    def __init__(self):
        #原始训练及测试集
        self.train_df = []
        self.test_df = []
        #log1p后的训练样本标签
        self.y_train = []
        #数值变换后的训练及测试集
        self.x_train = []
        self.x_test = []

    def dataclean(self):
        #读入原始数据集
        self.train_df = pd.read_csv('input/train.csv',index_col=0)
        self.test_df = pd.read_csv('input/test.csv',index_col=0)
        #print(self.train_df.head())

        #把类别标签log1p后与原始标签对比
        #prices = pd.DataFrame({'price':self.train_df['SalePrice'],'log(price+1)':np.log1p(self.train_df['SalePrice'])})
        #prices.hist()
        #plt.show()

        #类别标签log1p
        self.y_train = np.log1p(self.train_df.pop('SalePrice'))
        #把训练集与测试集合并进行预处理
        all_df = pd.concat((self.train_df,self.test_df),axis=0)
        #print(all_df.shape)

        #数据解释中把MSSubaClass定义为category类别，但查看数据集它就是
        #数字，如果直接进行运算那就出错了，把它变成string
        all_df['MSSubClass'] = all_df['MSSubClass'].astype(str)

        #用数值表示标签，需要one-hot独热编码
        #此刻MSSubClass被我们分成了12个column，每一个代表一个category。是就是1，不是就是0。
        #print(pd.get_dummies(all_df['MSSubClass'],prefix='MSSubClass').head())

        #把所有category数据都给one-hot
        all_dummy_df = pd.get_dummies(all_df)
        #print(all_dummy_df.head())

        #查看是否有某些列有缺失值
        #print(all_dummy_df.isnull().sum().sort_values(ascending=False).head(10))

        #计算各列均值，填充到缺失的位置
        mean_cols = all_dummy_df.mean()
        #print(mean_cols.head(10))
        all_dummy_df = all_dummy_df.fillna(mean_cols)
        #print(all_dummy_df.isnull().sum().sum())

        #把源数据放在标准分布内，使数据平滑
        numeric_cols = all_df.columns[all_df.dtypes != 'object']
        #print(numeric_cols)
        numeric_col_means = all_dummy_df.loc[:,numeric_cols].mean()
        numeric_col_std = all_dummy_df.loc[:,numeric_cols].std()
        all_dummy_df.loc[:,numeric_cols] = (all_dummy_df.loc[:,numeric_cols] - numeric_col_means)/numeric_col_std

        #数据集分回为训练集、测试集
        dummy_train_df = all_dummy_df.loc[self.train_df.index]
        dummy_test_df = all_dummy_df.loc[self.test_df.index]
        #print(dummy_train_df.shape,dummy_test_df.shape)

        #把数据集转化为Numpy Array
        self.x_train = dummy_train_df.values
        self.x_test = dummy_test_df.values

    def ridge(self):
        #生成等比数列基底默认10，幂（-3，2），生成50个
        alphas = np.logspace(-3,2,50)
        test_scores = []
        for alpha in alphas:
            clf = Ridge(alpha)
            test_score = np.sqrt(-cross_val_score(clf,self.x_train,self.y_train,cv=10,scoring='neg_mean_squared_error'))
            test_scores.append(np.mean(test_score))

        plt.figure(1)
        plt.subplot(331)
        plt.plot(alphas,test_scores)
        plt.title('Ridge Error')
        #plt.show()

    def forest(self):
        max_features = [.1,.3,.5,.7,.9,.99]
        test_scores = []
        for max_feat in max_features:
            clf = RandomForestRegressor(n_estimators=200,max_features=max_feat)
            test_score = np.sqrt(-cross_val_score(clf,self.x_train,self.y_train,cv=5,scoring='neg_mean_squared_error'))
            test_scores.append(np.mean(test_score))
        plt.subplot(332)
        plt.plot(max_features,test_scores)
        plt.title('Forest Error')
        #plt.show()

    def ensemble(self):
        rg = Ridge(alpha=15)
        rf = RandomForestRegressor(n_estimators=500,max_features=3)

        rg.fit(self.x_train,self.y_train)
        rf.fit(self.x_train,self.y_train)

        y_rg = np.expm1(rg.predict(self.x_test))
        y_rf = np.expm1(rf.predict(self.x_test))

        y_final = (y_rg+y_rf)/2

        submission_df = pd.DataFrame(data={'Id':self.test_df.index,'SalePrice':y_final})
        print(submission_df.head(10))

    #基于调参后岭回归模型调用bagging方法
    def baggingridge(self):
        rg = Ridge(15)
        params = [1,10,15,20,25,30,40]
        test_scores = []
        for param in params:
            clf = BaggingRegressor(n_estimators=param,base_estimator=rg)
            test_score = np.sqrt(-cross_val_score(clf,self.x_train,self.y_train,cv=10,scoring='neg_mean_squared_error'))
            test_scores.append(np.mean(test_score))
        plt.subplot(333)
        plt.plot(params,test_scores)
        plt.title('baggingridge Error')
        #plt.show()

    #bagging自带的DecisionTree
    def baggingdecisiontree(self):
        params = [10,15,20,25,30,40,50,60,70,100]
        test_scores = []
        for param in params:
            clf = BaggingRegressor(n_estimators=param)
            test_score = np.sqrt(-cross_val_score(clf,self.x_train,self.y_train,cv=10,scoring='neg_mean_squared_error'))
            test_scores.append(np.mean(test_score))
        plt.subplot(334)
        plt.plot(params,test_scores)
        plt.title('baggingdecisiontree Error')
        #plt.show()

    #基于调参后岭回归的boosting
    def boostingridge(self):
        rg = Ridge(15)
        params = [10,15,20,25,30,35,40,45,50]
        test_scores = []
        for param in params:
            clf = AdaBoostRegressor(n_estimators=param, base_estimator=rg)
            test_score = np.sqrt(-cross_val_score(clf, self.x_train, self.y_train, cv=10, scoring='neg_mean_squared_error'))
            test_scores.append(np.mean(test_score))
        plt.subplot(335)
        plt.plot(params, test_scores)
        plt.title('boostingridge Error')
        #plt.show()

    #Adaboost自带的DecisionTree
    def boostingdecisiontree(self):
        params = [10, 15, 20, 25, 30, 40, 50, 60, 70, 100]
        test_scores = []
        for param in params:
            clf = AdaBoostRegressor(n_estimators=param)
            test_score = np.sqrt(-cross_val_score(clf, self.x_train, self.y_train, cv=10, scoring='neg_mean_squared_error'))
            test_scores.append(np.mean(test_score))
        plt.subplot(336)
        plt.plot(params, test_scores)
        plt.title('boostingdecisiontree Error')
        #plt.show()

    def xgboost(self):
        params = [1,2,3,4,5,6]
        test_scores = []
        for param in params:
            clf = XGBRegressor(max_depth=param)
            test_score = np.sqrt(
                -cross_val_score(clf, self.x_train, self.y_train, cv=10, scoring='neg_mean_squared_error'))
            test_scores.append(np.mean(test_score))
        plt.subplot(337)
        plt.plot(params, test_scores)
        plt.title('xgboost Error')
        #plt.show()

if __name__ == '__main__':
    houseprice = Houseprice()
    #数据预处理
    houseprice.dataclean()
    #岭回归模型调参
    houseprice.ridge()
    #随机森林模型调参
    houseprice.forest()
    #集成调参后的岭回归和随机森林预测测试集
    #houseprice.ensemble()
    houseprice.baggingridge()
    houseprice.baggingdecisiontree()
    houseprice.boostingdecisiontree()
    houseprice.boostingridge()
    #xgboost
    houseprice.xgboost()
    plt.show()

