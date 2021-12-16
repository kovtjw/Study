from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM,Conv1D, Flatten, Dropout
from sklearn.datasets import load_diabetes
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import time


#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, random_state=42
)
x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],1) 
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],1)
print(x_train.shape,x_test.shape) # (309, 10) (133, 10)


#2. 모델구성
model = Sequential()
model.add(Conv1D(32, 2, activation='relu',input_shape = (10,1)))  
model.add(Dense(128, activation='relu'))      
model.add(Dense(64, activation='relu'))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(1))

#3. 컴파일, 훈련 
start = time.time()
model.compile(loss = 'mae', optimizer = 'adam')   # optimizer는 loss값을 최적화 한다.
model.fit(x_train, y_train, epochs = 100)
end = time.time()- start


#4. 평가, 예측 
loss = model.evaluate(x_test, y_test)
print('loss:', loss)

y_predict = model.predict(x_test)

print("걸린시간 : ", round(end, 3), '초')

'''
loss: 56.61195755004883
걸린시간 :  1.912 초
'''