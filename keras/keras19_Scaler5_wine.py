from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
import numpy as np


#1. 데이터

datasets = load_wine()
x = datasets.data
y = datasets.target
from tensorflow.keras.utils import to_categorical
y = to_categorical(y)

x_train, x_test, y_train, y_test = train_test_split(x,y,
        train_size =0.8, shuffle=True, random_state = 36)

scaler = MaxAbsScaler()         
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델구성
model = Sequential()
model.add(Dense(233, input_dim=13)) 
model.add(Dense(144,activation='relu'))
model.add(Dense(55))
model.add(Dense(21))
model.add(Dense(3, activation= 'softmax'))

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])
 
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=10, mode='auto',
                   verbose=1, restore_best_weights=False) # restore_best_weights=True : 최종값 이전에 가장 좋았던 값 도출함

model.fit(x_train, y_train, epochs=100, batch_size=5,
          validation_split=0.3, callbacks=[es])

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss :', loss[0])
print('accuracy :', loss[1])




'''
# 결과 
그냥 : loss : 0.4261055290699005

MinMax : loss : 0.06908071041107178
 
Standard : loss : 0.09674564003944397

Robuster : loss : 0.12121929228305817

MaxAbs : loss : 0.0913960188627243
'''

'''
# relu 결과 

그냥 : loss : 0.24319495260715485

MinMax : loss : 0.03654302656650543
 
Standard : loss : 0.18639662861824036

Robuster : loss : 0.07303806394338608

MaxAbs : loss : 0.07150081545114517

'''