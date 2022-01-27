import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import QuantileTransformer, PowerTransformer
from sklearn.datasets import fetch_covtype
from imblearn.over_sampling import SMOTE
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import accuracy_score, f1_score
import time

#1. 데이터
datasets = fetch_covtype()
x = datasets.data
y = datasets.target
print(x.shape, y.shape) 
print(pd.Series(y).value_counts())

'''
2    283301
1    211840
3     35754
7     20510
6     17367
5      9493
4      2747
'''
x_train, x_test, y_train, y_test = train_test_split(x,y,
        train_size =0.8, shuffle=True, random_state = 66, stratify= y)  # stratify = y >> yes가 아니고, y(target)이다.
import pickle
path = './_save/'
x_train, y_train = pickle.load(open(path + 'm30_smote_save.dat','rb'))
start = time.time()
smote = SMOTE(random_state=66, k_neighbors=2)
x_train, y_train = smote.fit_resample(x_train, y_train)
smote_data1 = x_train, y_train 

end = time.time()
model = XGBClassifier(n_jobs = -1)
model.fit(x_train, y_train)
score = model.score(x_test, y_test)
print('model.score :', round(score,4))
y_pred = model.predict(x_test)
print('accuracy_score :', round(accuracy_score(y_test, y_pred),4))
f1 = f1_score(y_test, y_pred, average='macro')
print('f1_score:',f1)
print('걸린시간 : ', end-start)

