import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, Flatten

x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],
              [5,6,7],[6,7,8],[7,8,9],[8,9,10],
              [9,10,11],[10,11,12],
              [20,30,40],[30,40,50],[40,50,60]])
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])
x_predict = np.array([50,60,70])
size = 4


def split_x(data, size):
    aaa = []
    for i in range(len(data) - size +1): # 1~10 - 5 +1 //len은 0부터 시작하기 때문에 +1을 해주어야 함
        subset = data[i : (i+size)]      # [1 : 6] = 1,2,3,4,5 
        aaa.append(subset)                  # 1,2,3,4,5에 []를 붙인다.
    return np.array(aaa)                    # numpy 배열로 리턴한다.

x1 = split_x(x, size)                  # 대입했을 때
y1 = split_x(y, size)
print(x1.shape) # (9,5,3)
print(y1.shape)# (9,5, )


#2. 모델구성
model = Sequential()
model.add(LSTM(32,activation='relu',input_shape = (5,3))) # input_shape 할 때 행은 넣지 않는다. 
model.add(Flatten()) 
model.add(Dense(128, activation='relu'))       # 모든 activation은 다음으로 보내는 값을 한정 시키기 때문에 윗 줄에서도 가능하다.
model.add(Dense(64, activation='relu'))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(1))

#3. 컴파일, 훈련 
model.compile(loss = 'mae', optimizer = 'rmsprop')   # optimizer는 loss값을 최적화 한다.
model.fit(x1, y1, epochs = 2000)

#4. 평가, 예측 
model.evaluate(x1, y1)
result = model.predict([[[50],[60],[70]]])
print(result)
