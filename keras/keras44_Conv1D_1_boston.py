from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM,Conv1D, Flatten
from sklearn.datasets import load_boston
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import time

a = load_boston()
x = a.data
y = a.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, random_state=42
)

x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],1) #(354, 13)
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],1) #(152, 13)

print(x_train.shape,x_test.shape)

#2. 모델구성
model = Sequential()
# model.add(LSTM(32,activation='relu',input_shape = (13,1)))
model.add(Conv1D(10, 2, input_shape=(13,1)))
model.add(Flatten())  
model.add(Dense(128, activation='relu'))      
model.add(Dense(64, activation='relu'))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(1))

model.summary()

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
loss: 5.997178554534912
걸린시간 :  1.608 초
'''