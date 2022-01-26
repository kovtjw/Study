import pandas as pd 
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import time
from xgboost import XGBRegressor, XGBClassifier
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import QuantileTransformer, PowerTransformer
import matplotlib.pyplot as plt
#1. 데이터
path = 'D:\\_data\\'
datasets = pd.read_csv(path + 'winequality-white.csv',sep=';'
                       , header = 0)

count_data = datasets.groupby("quality")['quality'].count()  
print(count_data)

plt.bar(count_data.index, count_data)
plt.show()
x = datasets.drop("quality",axis=1)
y = datasets["quality"]

x_train, x_test, y_train, y_test = train_test_split(x,y,
        train_size =0.8, shuffle=True, random_state = 66, stratify= y)  # stratify = y >> yes가 아니고, y(target)이다.

scaler = PowerTransformer()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델
model = XGBClassifier(
    n_jobs = -1,  
    n_estimators = 100,
    learning_rate = 0.054,
    # subsample_for_bin= 200000,
    max_depth = 6,
    min_child_weight = 1,
    subsample = 1,
    colsample_bytree = 1,
    reg_alpha = 1,              # 규제 : L1 = lasso
)
#3. 훈련
start = time.time()
model.fit(x_train, y_train, verbose = 3,
        #   eval_set=[(x_train,y_train),(x_test, y_test)],
          eval_metric='merror',              #rmse, mae, logloss, error
        #   early_stopping_rounds=2000,
          )
end = time.time()

result = model.score(x_test, y_test)
print('results :', round(result,4))

y_pred = model.predict(x_test)
acc = accuracy_score(y_test, y_pred)
print('r2 :',round(acc,4))
print('걸린 시간 :', round(end-start, 4))