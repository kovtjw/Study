import numpy as np 
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler

#1. 데이터
datasets = fetch_covtype()
x = datasets.data 
y = datasets.target

import pandas as pd 
y = pd.get_dummies(y)
print(x.shape,y.shape)

x_train, x_test, y_train, y_test = train_test_split(x,y,
        train_size =0.8, shuffle=True, random_state = 42)


scaler = MaxAbsScaler()         
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)



#2. 모델구성
model = Sequential()
model.add(Dense(100, input_dim=54)) 
model.add(Dense(80, activation='relu'))
model.add(Dropout(0.2)) 
model.add(Dense(130, activation='sigmoid'))
model.add(Dropout(0.3)) 
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
model.fit(x_train, y_train, epochs =100, batch_size=1000,
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



print ('====================== 2. load_model 출력 ========================')
model2 = load_model('./_save/keras27_6_save_model.h5')

loss2 = model2.evaluate(x_test, y_test)
print('loss:', loss2)

y_predict2 = model2.predict(x_test)
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict2)
print('r2 스코어:', r2)

print ('====================== 3. ModelCheckPoint 출력 ========================')
model3 = load_model('./_ModelCheckPoint/keras27_6_MCP.hdf5')

loss3 = model3.evaluate(x_test, y_test)
print('loss:', loss3)

y_predict3 = model3.predict(x_test)
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2 스코어:', r2)
'''
====================== 1. 기본출력 ========================
3632/3632 [==============================] - 2s 464us/step - loss: 0.2706 - accuracy: 0.8902
loss: [0.27058833837509155, 0.8902437686920166]
r2 스코어: 0.7204862756974876
====================== 2. load_model 출력 ========================
3632/3632 [==============================] - 2s 463us/step - loss: 0.2706 - accuracy: 0.8902
loss: [0.27058833837509155, 0.8902437686920166]
r2 스코어: 0.7204862756974876
====================== 3. ModelCheckPoint 출력 ========================
3632/3632 [==============================] - 2s 470us/step - loss: 0.2706 - accuracy: 0.8902
loss: [0.27058833837509155, 0.8902437686920166]
r2 스코어: 0.7204862756974876

'''
###########################드랍아웃 적용 시#############################

'''
====================== 1. 기본출력 ========================
3632/3632 [==============================] - 2s 482us/step - loss: 0.3420 - accuracy: 0.8607
loss: [0.3419801592826843, 0.8606920838356018]
r2 스코어: 0.6634871784380266
====================== 2. load_model 출력 ========================
3632/3632 [==============================] - 2s 470us/step - loss: 0.3420 - accuracy: 0.8607
loss: [0.3419801592826843, 0.8606920838356018]
r2 스코어: 0.6634871784380266
====================== 3. ModelCheckPoint 출력 ========================
3632/3632 [==============================] - 2s 467us/step - loss: 0.3364 - accuracy: 0.8641
loss: [0.33635151386260986, 0.8640740513801575]
r2 스코어: 0.6634871784380266

'''