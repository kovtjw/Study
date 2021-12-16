import numpy as np 
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D
import time
#1. 데이터
datasets = fetch_covtype()
x = datasets.data 
y = datasets.target


print(x.shape,y.shape)
x_train, x_test, y_train, y_test = train_test_split(x,y,
        train_size =0.8, shuffle=True, random_state = 42)
x_train = x_train.reshape(464809, 54,1)
x_test = x_test.reshape(116203, 54,1)

#2. 모델구성
model = Sequential()
model.add(Conv1D(32,2,activation='relu',input_shape = (54,1))) 
model.add(Dense(80, activation='relu'))
model.add(Dense(130, activation='sigmoid'))
model.add(Dense(80, activation='relu'))
model.add(Dense(5))
model.add(Dense(1, activation = 'softmax'))

#3. 컴파일, 훈련
start = time.time()

model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

model.fit(x_train, y_train, epochs =20, batch_size=1000,
          validation_split=0.3)
end = time.time()- start

#4. 평가, 예측
print ('====================== 1. 기본출력 ========================')
loss = model.evaluate(x_test, y_test)
print('loss:', loss)

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2 스코어:', r2)

print("걸린시간 : ", round(end, 3), '초')
