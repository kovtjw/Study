import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, Bidirectional ,Conv1D, Flatten

#1. 데이터
x = np.array([[1,2,3],
             [2,3,4],
             [3,4,5],
             [4,5,6]])
y = np.array([4,5,6,7])

print(x.shape,y.shape)  # (4, 3) (4,)

x = x.reshape(4, 3, 1) 

#2. 모델구성
model = Sequential()
# model.add(Bidirectional(SimpleRNN(10),input_shape = (3,1))) # 파라미터 연산이 2배가 된다.
model.add(Conv1D(10, 2, input_shape=(3,1)))
model.add(Flatten())
model.add(Dense(16, activation='relu'))      
model.add(Dense(8))
model.add(Dense(4))
model.add(Dense(1))

model.summary()

#3. 컴파일, 훈련 

model.compile(loss = 'mae', optimizer = 'adam')   # optimizer는 loss값을 최적화 한다.
model.fit(x, y, epochs = 300)


#4. 평가, 예측 
model.evaluate(x, y)
result = model.predict([[[5],[6],[7]]])
print(result)


