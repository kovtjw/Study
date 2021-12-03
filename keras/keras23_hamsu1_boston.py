from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
import numpy as np

#1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets.target
# print(np.min(x), np.max(x)) # 0.0 711.0


# print(x.shape)      #(506, 13): 컬럼이 13개
# x = x/711.              # .을 쓰는 이유는 부동소수점으로 나눈다는 뜻
# x = x/np.max(x)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, random_state=46
)

scaler =  MaxAbsScaler()         
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델구성
input1 = Input(shape=(13,))
dense1 = Dense(100)(input1)
dense2 = Dense(80)(dense1)
dense3 = Dense(130)(dense2)
dense4 = Dense(80)(dense3)
dense5 = Dense(5)(dense4)
output1 = Dense(1)(dense5)
model = Model(inputs=input1, outputs = output1)

'''
model = Sequential()
model.add(Dense(100, input_dim=10)) 
model.add(Dense(80,activation='relu'))
model.add(Dense(130))
model.add(Dense(80))
model.add(Dense(5))
model.add(Dense(1))
'''
#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=30, mode='min', verbose=1) 

model.fit(x_train, y_train, epochs=10000, batch_size=1,
          validation_split=0.3, callbacks=[es])

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss:', loss)

y_predict = model.predict(x_test)


'''
# scaler 결과 
그냥 : loss: 19.14067268371582

MinMax : loss: 16.720535278320312
 
Standard : loss: 17.38316535949707

Robuster : loss: 15.025249481201172

MaxAbs : loss: 20.80170440673828
'''

'''
# relu 결과 

그냥 : loss: 15.611274719238281

MinMax : loss: 8.855862617492676
 
Standard : loss: 12.234639167785645

Robuster :loss: 9.609686851501465

MaxAbs : loss: 10.25717544555664

'''