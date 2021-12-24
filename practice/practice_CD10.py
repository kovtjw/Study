import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings(action='ignore')
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from lightgbm import LGBMClassifier,plot_importance

# 1. 데이터

path = '../_data/dacon/cardiovascular disease/'
train = pd.read_csv(path+'train.csv')
test_file = pd.read_csv(path+'test.csv')
submit_file = pd.read_csv(path + 'sample_submission.csv')

features=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
       'thalach', 'exang', 'oldpeak', 'slope', 'ca']
target='target'

X_train=train[features]
Y_train=train[target]
X_test=test_file[features]

lr=LogisticRegression()
dtc=DecisionTreeClassifier()
lgb=LGBMClassifier()

cross_val_score(lr,X_train,Y_train,cv=5).mean()
cross_val_score(dtc,X_train,Y_train,cv=5).mean()
cross_val_score(lgb,X_train,Y_train,cv=5).mean()
lgb.fit(X_train,Y_train)
lr.fit(X_train,Y_train)
result=lr.predict(X_test)
submit_file['target']=result
submit_file.to_csv('submission.csv',index=False)