from tensorflow.keras.layers import Dense, LSTM, Conv1D
from sklearn.datasets import load_wine
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
import numpy as np
import time


#1. 데이터

datasets = load_wine()
x = datasets.data
y = datasets.target


x_train, x_test, y_train, y_test = train_test_split(x,y,
        train_size =0.8, shuffle=True, random_state = 42)
x_train = x_train.reshape(142,13,1)
x_test = x_test.reshape(36,13,1)

#2. 모델구성
model = Sequential()
model.add(Conv1D(32,2,input_shape = (13,1)))  
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

'''
loss: 0.5542266368865967
걸린시간 :  1.436 초
'''