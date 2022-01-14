from tensorflow.keras.layers import Dense, LSTM
from sklearn.datasets import load_wine
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
import numpy as np
import time
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.model_selection import cross_validate, KFold
from sklearn.utils import shuffle
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

#1. 데이터

datasets = load_wine()
x = datasets.data
y = datasets.target
from tensorflow.keras.utils import to_categorical
y = to_categorical(y)

fold_df_clf = RandomForestClassifier()
kfold = KFold(n_splits=5)
cv_accuracy = []

n_iter = 0 
for train_idx, test_idx in kfold.split(x):
    x_train, x_test = x[train_idx], x[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    fold_df_clf.fit(x_train, y_train)
    fold_pred = fold_df_clf.predict(x_test)

    n_iter += 1
    accuracy = np.round(accuracy_score(y_test,fold_pred),4)
    print('\n{} 교차검증정확도:{}, 학습데이터 크기: {}, 검증데이터 크기 : {}'.format(n_iter, accuracy, x_train.shape[0], x_test.shape[0]))
    cv_accuracy.append(accuracy)
print('\n')
print('\n 평균검증 정확도 :', np.mean(cv_accuracy)) 

#2. 모델구성
model = Sequential()
model.add(LSTM(32,activation='relu',input_shape = (13,1)))  
model.add(Dense(128, activation='relu'))      
model.add(Dense(64, activation='relu'))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(1))

#3. 컴파일, 훈련 
start = time.time()
model.compile(loss = 'mse', optimizer = 'adam')   # optimizer는 loss값을 최적화 한다.
model.fit(x_train, y_train, epochs = 100)
end = time.time()- start

#4. 평가, 예측 
loss = model.evaluate(x_test, y_test)
print('loss:', loss)

y_predict = model.predict(x_test)
print("걸린시간 : ", round(end, 3), '초')