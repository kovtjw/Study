import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from pandas import set_option 
from pandas.plotting import scatter_matrix 
from sklearn.model_selection import train_test_split,KFold,cross_val_score,GridSearchCV 
from sklearn.linear_model import LinearRegression, LogisticRegression, Lasso, ElasticNet 
from sklearn.tree import DecisionTreeRegressor 
from sklearn.neighbors import KNeighborsRegressor 
from sklearn.svm import SVR 
from sklearn.pipeline import Pipeline 
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, AdaBoostRegressor 
from sklearn.metrics import mean_squared_error 
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis 
from sklearn.naive_bayes import GaussianNB 
from sklearn import preprocessing 
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler, MaxAbsScaler 
import seaborn as sns 
 
#1. 데이터 
path = '../_data/project data/' 
gwangju = pd.read_csv(path +"gwangju .csv")
x = gwangju.drop(['일자','가격'], axis = 1)
y = gwangju['가격']


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=42) 

model1 = RandomForestRegressor(n_estimators = 100, max_depth=10, min_samples_split=40, min_samples_leaf =30) 
model2 = GradientBoostingRegressor(n_estimators = 100, learning_rate = 0.3, max_depth=10, min_samples_split=40, min_samples_leaf =30) 
model3 = ExtraTreesRegressor(n_estimators=100, max_depth=16, random_state=7) 
model4 = AdaBoostRegressor(n_estimators=100, random_state=7) 
model5 = XGBRegressor(n_estimators = 100, learning_rate = 0.3, max_depth=10, min_samples_split=40, min_samples_leaf =30) 
model6 = LGBMRegressor(n_estimators = 100, learning_rate = 0.1, max_depth=10, min_samples_split=40, min_samples_leaf =30) 
model7 = CatBoostRegressor(n_estimators=100, max_depth=16, random_state=7) 
 
from sklearn.ensemble import VotingClassifier 
voting_model = VotingClassifier(estimators=[('RandomForestRegressor', model1), 
                                            ('GradientBoostingRegressor', model2), 
                                            ('ExtraTreesRegressor', model3), 
                                            ('AdaBoostRegressor', model4), 
                                            ('XGBRegressor', model5), 
                                            ('LGBMRegressor', model6), 
                                            ('CatBoostRegressor', model7)], voting='hard') 
 
classifiers = [model1,model2,model3,model4,model5,model6,model7] 
from sklearn.metrics import r2_score 
 
for classifier in classifiers: 
    classifier.fit(x_train, y_train) 
    y_predict = classifier.predict(x_test) 
    r2 = r2_score(y_test, y_predict) 
           
    class_name = classifier.__class__.__name__ 
    print("============== " + class_name + " ==================") 
     
    print('r2 스코어 : ', round(r2,3)) 
    print('예측값 : ', y_predict[-1]) 
 
