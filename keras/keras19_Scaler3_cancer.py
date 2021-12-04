from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
import numpy as np

#1. 데이터
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=42
)


scaler = MaxAbsScaler()         
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


#2. 모델구성
model = Sequential()
model.add(Dense(100, input_dim=30)) 
model.add(Dense(80,activation='relu'))
model.add(Dense(130,activation='relu'))
model.add(Dense(80))
model.add(Dense(5))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=100, mode='min', verbose=1) 
model.fit(x_train, y_train, epochs=1000, batch_size=32,
          validation_split=0.3, callbacks=[es])
 
model.save('./_save/keras19_3_save_model.h5')

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss:', loss)


'''
# 결과 
그냥 : loss: 0.3905664086341858

MinMax : loss: 0.06161541864275932
 
Standard : loss: 0.0632854774594307

Robuster : loss: 0.0608515702188015

MaxAbs : loss: 0.06307485699653625 
'''

'''
# relu 결과 

그냥 : loss: 0.23272587358951569

MinMax : loss: 0.03191361576318741
 
Standard : loss: 0.02117059752345085

Robuster : loss: 0.014400756917893887

MaxAbs : loss: 0.032459620386362076

'''