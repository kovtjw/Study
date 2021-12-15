import numpy as np 
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

#1. 데이터
datasets = fetch_covtype()
x = datasets.data 
y = datasets.target
print(y.shape) # (581012,)

import pandas as pd 
y = pd.get_dummies(y)
print(x.shape,y.shape) #(581012, 54) (581012, 7)

x_train, x_test, y_train, y_test = train_test_split(x,y,
        train_size =0.8, shuffle=True, random_state = 42)

# print(x_train.shape,x_test.shape)  # (464809, 54) (116203, 54)
x_train = x_train.reshape(464809, 54,1)
x_test = x_test.reshape(116203, 54,1)
# y = y.reshape(581012, 7,1)

print(x_train.shape,x_test.shape)

#2. 모델구성
model = Sequential()
model.add(LSTM(32,activation='relu',input_shape = (54,1))) 
model.add(Dense(80, activation='relu'))
model.add(Dense(130, activation='sigmoid'))
model.add(Dense(80, activation='relu'))
model.add(Dense(5))
model.add(Dense(7, activation = 'softmax'))

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
es = EarlyStopping(monitor = 'val_loss', patience=10, mode='auto',
                   verbose =1, restore_best_weights=True)
mcp = ModelCheckpoint (monitor = 'val_loss', mode = 'min', verbose = 1, save_best_only=True,
                       filepath = './_ModelCheckPoint/keras27_6_MCP.hdf5')
model.fit(x_train, y_train, epochs =20, batch_size=1000,
          validation_split=0.3, callbacks=[es,mcp])

model.save('./_save/keras27_6_save_model.h5')


#4. 평가, 예측
print ('====================== 1. 기본출력 ========================')
loss = model.evaluate(x_test, y_test)
print('loss:', loss)

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2 스코어:', r2)

