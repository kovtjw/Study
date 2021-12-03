import numpy as np 
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense , Input
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler

#1. 데이터
datasets = fetch_covtype()
x = datasets.data 
y = datasets.target

import pandas as pd 
y = pd.get_dummies(y)
print(x.shape,y.shape)

x_train, x_test, y_train, y_test = train_test_split(x,y,
        train_size =0.8, shuffle=True, random_state = 36)


scaler = MaxAbsScaler()         
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)



#2. 모델구성
input1 = Input(shape=(54,))
dense1 = Dense(100)(input1)
dense2 = Dense(80)(dense1)
dense3 = Dense(130)(dense2)
dense4 = Dense(80)(dense3)
dense5 = Dense(5)(dense4)
output1 = Dense(1)(dense5)
model = Model(inputs=input1, outputs = output1)

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor = 'val_loss', patience=10, mode='auto',
                   verbose =1, restore_best_weights=False)

model.fit(x_train, y_train, epochs =100, batch_size=1000,
          validation_split=0.3, callbacks=[es])

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss :', loss[0])
print('accuracy : ', loss[1])
