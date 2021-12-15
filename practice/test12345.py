from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.datasets import load_boston
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

a = load_boston()
x = a.data
y = a.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, random_state=42
)
print(x_train.shape,x_test.shape)

x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],1) #(354, 13)
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],1) #(152, 13)


#2. 모델구성
model = Sequential()
model.add(LSTM(32,activation='relu',input_shape = (13,1)))  
model.add(Dense(128, activation='relu'))      
model.add(Dense(64, activation='relu'))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(1))

model.summary()

#3. 컴파일, 훈련 
model.compile(loss = 'mae', optimizer = 'adam')   # optimizer는 loss값을 최적화 한다.
model.fit(x_train, y_train, epochs = 100)

#4. 평가, 예측 
loss = model.evaluate(x_test, y_test)
print('loss:', loss)

y_predict = model.predict(x_test)
