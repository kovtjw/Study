from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.datasets import load_diabetes
import numpy as np
from sklearn.model_selection import train_test_split

a = load_diabetes()
b = a.data
c = a.target

size = 7

def split_x(data, size):
    aaa = []
    for i in range(len(data) - size +1): # 1~10 - 5 +1 //size가 나오는 가장 마지막 것을 생각해서 +1을 해주어야 함
        subset = data[i : (i+size)]      # [1 : 6] = 1,2,3,4,5 
        aaa.append(subset)                  # 1,2,3,4,5에 []를 붙인다.
    return np.array(aaa)

x = split_x(b,size)
y = split_x(c,size)
print(x.shape)   # (436, 7, 10)
print(y.shape)   # (436, 7)
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, random_state=42
)
print(x_train.shape) # (305, 7, 10)
print(y_test.shape) # (131, 7)



#2. 모델구성
model = Sequential()
model.add(LSTM(32,activation='relu',input_shape = (7,10)))  
model.add(Dense(128, activation='relu'))      
model.add(Dense(64, activation='relu'))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(1))


#3. 컴파일, 훈련 
model.compile(loss = 'mae', optimizer = 'adam')   # optimizer는 loss값을 최적화 한다.
model.fit(x, y, epochs = 100)

#4. 평가, 예측 
loss = model.evaluate(x_test, y_test)
print('loss:', loss)

y_predict = model.predict(x_test)