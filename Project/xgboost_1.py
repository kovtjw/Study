import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler,MaxAbsScaler, LabelEncoder
from sklearn.metrics import f1_score
import xgboost as xgb
from xgboost import XGBRFClassifier,XGBClassifier, XGBRegressor
from xgboost import plot_importance, plot_tree
import graphviz
import matplotlib.pyplot as plt
import xgboost
from xgboost.core import Booster
plt.style.use(['seaborn-whitegrid'])
from sklearn.metrics import mean_squared_error as MSE

path = '../_data/project data/' 
gwangju = pd.read_csv(path +"광주 .csv")
x = gwangju.drop(['일자','가격'], axis = 1)
y = gwangju['가격']
x = np.array(x)
print(plt.plot(x))



x_train,x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model_xgboost = xgboost.XGBClassifier(learning_rate = 0.1,      # 학습률
                                      max_depth = 5,            # 각 트리의 최대 깊이 
                                      n_estimators = 200,       # 200개의 나무
                                      subsamples = 0.5,         # 하위 샘플 훈련 데이터 세트의 관측치의 50%가 무작위로 선택 되어 각 개별 트리를 생성한다는 의미 >> 빠르게 훈련하는데 도움되고, 하위 샘플이 과적합 되는 것을 방지
                                      colsample_bytree = 0.5,   # 트리별 호출
                                      eval_metric = 'auc',
                                      verbosity = 1)

model_xgboost = xgboost.XGBRegressor(base_score = 0.5,      # 학습률
                                      booster ='gbtree',
                                      max_depth = 5, # 각 트리의 최대 깊이 
                                      n_estimators = 200,       # 200개의 나무
                                         # 하위 샘플 훈련 데이터 세트의 관측치의 50%가 무작위로 선택 되어 각 개별 트리를 생성한다는 의미 >> 빠르게 훈련하는데 도움되고, 하위 샘플이 과적합 되는 것을 방지
                                      colsample_bylevel = 1,   # 트리별 호출
                                      eval_metric = 'auc',
                                      verbosity = 1)
eval_set = [(x_test,y_test)]

model_xgboost.fit(x_train,
                  y_train,
                  early_stopping_rounds=10,
                  eval_set = eval_set,
                  verbose = True)
xgb_r = xgb.XGBRegressor(objective='reg:linear',
                         n_estimators = 10, seed = 123)
xgb_r.fit(x_train,y_train)

pred = xgb_r.predict(x_test)

rmse = np.sqrt(MSE(y_test, pred))
print("RMSE : % f" %(rmse))

print(pred)

fig, ax = plt.subplots(figsize=(10,12))
plot_importance(xgb_r, ax=ax)

# # 

xgboost.plot_importance(model_xgboost)
plt.show()
