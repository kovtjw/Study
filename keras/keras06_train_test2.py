from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np


#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])

### 과제 ###
# train과 test 비율을 8:2으로 분리하시오. >> 리스트의 슬라이싱으로 자르기  

x_train = x[:8]
x_test = x[8:]
y_train = y[:8]
y_test = y[8:]

### 완료 후 모델>평가,예측까지###
#2. 모델구성
model = Sequential()
model.add(Dense(100, input_dim=1)) 
model.add(Dense(80))
model.add(Dense(130))
model.add(Dense(80))
model.add(Dense(5))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse',optimizer = 'adam')
model.fit(x_train, y_train, epochs=100, batch_size=1) 

#4. 평가, 예측
loss = model.evaluate(x_test,y_test)
print('loss: ', loss)
result = model.predict([114])
print('11의 예측값 :', result)

