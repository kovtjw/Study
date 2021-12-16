from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D
from tensorflow.keras.utils import to_categorical
import numpy as np
import time

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.reshape(50000,32,96)
x_test = x_test.reshape(10000,32,96)
y_train = to_categorical(y_train) # (50000, 32, 32, 3)
y_test = to_categorical(y_test)

#2. 모델구성
model = Sequential()
model.add(Conv1D(32,2,activation='relu',input_shape = (32,96)))  
model.add(Dense(128, activation='relu'))      
model.add(Dense(64, activation='relu'))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(10, activation = 'softmax'))

#3. 컴파일, 훈련
start = time.time()

model.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=16, batch_size=16,
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

