import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, Dropout 
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler

#1. 데이터
x = np.array([[1,2,3],
             [2,3,4],
             [3,4,5],
             [4,5,6]])
y = np.array([4,5,6,7])

print(x.shape,y.shape)  # (4, 3) (4,)


# input_shape = (batch_size, timesteps, feature)
# input_shape = (행, 열, 몇 개씩 자르는지!!)  >>> 2차원 데이터를 3차원으로 변경해주어야 한다. >>>> reshape는 데이터의 내용과 순서를 바꾸지 않는다.
x = x.reshape(4, 3, 1) 
# print(x)


#2. 모델구성
model = Sequential()
model.add(SimpleRNN(32,activation='relu',input_shape = (3,1))) # input_shape 할 때 행은 넣지 않는다. 
model.add(Dense(16, activation='relu'))       # 모든 activation은 다음으로 보내는 값을 한정 시키기 때문에 윗 줄에서도 가능하다.
model.add(Dense(8))
model.add(Dense(4))
model.add(Dense(1))



#3. 컴파일, 훈련 
model.compile(loss = 'mae', optimizer = 'adam')   # optimizer는 loss값을 최적화 한다.
model.fit(x, y, epochs = 300)

#4. 평가, 예측 
model.evaluate(x, y)
result = model.predict([[[5],[6],[7]]])
print(result)
