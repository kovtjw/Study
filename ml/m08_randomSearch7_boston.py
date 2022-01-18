import numpy as np
from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor 
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, r2_score
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# 실습  // 모델 : RandomForestClassifier


#1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets.target


x_train, x_test, y_train, y_test = train_test_split(x,y,
        shuffle= True, random_state=66, train_size = 0.8)
n_splits = 5
kfold = KFold(n_splits = n_splits, shuffle = True, random_state=100)

parameters = [
    {'n_estimators': [100,200,300],'max_depth' : [6,8,10,12]},
    {'min_samples_leaf' :[3,5,7,10],'min_samples_split' : [2,3,5,10]}]

#2. 모델 구성
model = RandomizedSearchCV(RandomForestRegressor(), parameters, cv = kfold, verbose = 3,
                     refit=True, n_jobs=1)

#3. 훈련
import time
start = time.time()
model.fit(x_train, y_train)
end = time.time()

#4. 평가, 예측
print('최적의 매개 변수 :', model.best_estimator_)
print('최적의 파라미터 : ', model.best_params_)
print('best_score_ :', model.best_score_)
print('소요 시간 :', end-start)

'''
최적의 매개 변수 : RandomForestRegressor(max_depth=12)
최적의 파라미터 :  {'n_estimators': 100, 'max_depth': 12}
best_score_ : 0.8498099759507302
소요 시간 : 10.277530193328857

최적의 매개 변수 : XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
             colsample_bynode=1, colsample_bytree=1, enable_categorical=False,
             gamma=0, gpu_id=-1, importance_type=None,
             interaction_constraints='', learning_rate=0.300000012,
             max_delta_step=0, max_depth=6, min_child_weight=1, missing=nan,
             monotone_constraints='()', n_estimators=200, n_jobs=12,
             num_parallel_tree=1, predictor='auto', random_state=0, reg_alpha=0,
             reg_lambda=1, scale_pos_weight=1, subsample=1, tree_method='exact',
             validate_parameters=1, verbosity=None)
최적의 파라미터 :  {'n_estimators': 200, 'max_depth': 6}
best_score_ : 0.853274730459266
소요 시간 : 7.457101583480835
'''