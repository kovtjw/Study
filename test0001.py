import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array([range(10), range(21,31), range(201,211)])
x = np.transpose(x) #(10,3)

y = np.array([[1,2,3,4,5,6,7,8,9,10],
             [1,1.1,1.2,1.3,1.4,1.5,
             1.6,1.5,1.4,1.3],
             [10,9,8,7,6,5,4,3,2,1]])
y = np.transpose(y)
print(y.shape)

#2. 모델 구성 

model = Sequential()
model.add(Dense(100, input_dim=3))
model.add(Dense(80))
model.add(Dense(130))
model.add(Dense(80))
model.add(Dense(5))
model.add(Dense(3))

#3. 컴파일, 훈련

model.compile(loss='mse', optimizer='adam')

model.fit(x,y, epochs=50, batch_size=3)

#4. 평가, 예측
loss = model.evaluate(x,y)
print('loss:',loss)
y_predict = model.predict([9,30,210])
print('[9, 30, 210]의 예측값:', y_predict)

