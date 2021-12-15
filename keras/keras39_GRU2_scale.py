import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM,GRU
import time

#1. 데이터
x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],
              [5,6,7],[6,7,8],[7,8,9],[8,9,10],
              [9,10,11],[10,11,12],
              [20,30,40],[30,40,50],[40,50,60]])
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])
x_predict = np.array([50,60,70])

print(x.shape,y.shape) # (13, 3) (13,)
x = x.reshape(13,3,1)
print(x.shape) # (13,3,1)
print(x_predict.shape) # (3,)

#2. 모델구성
model = Sequential()
model.add(LSTM(32,activation='relu',input_shape = (3,1))) # input_shape 할 때 행은 넣지 않는다. 
model.add(Dense(128, activation='relu'))       # 모든 activation은 다음으로 보내는 값을 한정 시키기 때문에 윗 줄에서도 가능하다.
model.add(Dense(64, activation='relu'))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(1))

#3. 컴파일, 훈련 
start = time.time()
model.compile(loss = 'mae', optimizer = 'adam')   # optimizer는 loss값을 최적화 한다.
model.fit(x, y, epochs = 600)
end = time.time() - start

print('걸린시간:', round(end,3), '초')

#4. 평가, 예측 
model.evaluate(x, y)
result = model.predict([[[50],[60],[70]]])
print(result)


'''
fit에 time 걸어서 속도 확인하기

'''