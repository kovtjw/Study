from tensorflow.keras.layers import Dense, LSTM
from sklearn.datasets import load_wine
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
import numpy as np


#1. 데이터

datasets = load_wine()
x = datasets.data
y = datasets.target
from tensorflow.keras.utils import to_categorical
y = to_categorical(y)

x_train, x_test, y_train, y_test = train_test_split(x,y,
        train_size =0.8, shuffle=True, random_state = 42)
x_train = x_train.reshape(142,13,1)
x_test = x_test.reshape(36,13,1)
print(x_train.shape,x_test.shape)


#2. 모델구성
model = Sequential()
model.add(LSTM(32,activation='relu',input_shape = (13,1)))  
model.add(Dense(128, activation='relu'))      
model.add(Dense(64, activation='relu'))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(1))

#3. 컴파일, 훈련 
model.compile(loss = 'mae', optimizer = 'adam')   # optimizer는 loss값을 최적화 한다.
model.fit(x_train, y_train, epochs = 100)

#4. 평가, 예측 
loss = model.evaluate(x_test, y_test)
print('loss:', loss)

y_predict = model.predict(x_test)
