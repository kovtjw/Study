from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
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
        train_size =0.8, shuffle=True, random_state = 42)

scaler = MinMaxScaler()         
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델구성
model = Sequential()
model.add(Dense(100, input_dim=13)) 
model.add(Dense(80, activation='relu'))
model.add(Dropout(0.2)) 
model.add(Dense(130, activation='sigmoid'))
model.add(Dropout(0.3)) 
model.add(Dense(80, activation='relu'))
model.add(Dense(5))
model.add(Dense(3, activation = 'softmax'))

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])
 
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', patience=100, mode='auto',
                   verbose=1, restore_best_weights=False) # restore_best_weights=True : 최종값 이전에 가장 좋았던 값 도출함
mcp = ModelCheckpoint (monitor = 'val_loss', mode = 'min', verbose = 1, save_best_only=True,
                       filepath = './_ModelCheckPoint/keras27_5_MCP.hdf5')
model.fit(x_train, y_train, epochs=1000, batch_size=32,
          validation_split=0.3, callbacks=[es,mcp])

model.save('./_save/keras27_5_save_model.h5')

#4. 평가, 예측
print ('====================== 1. 기본출력 ========================')
loss = model.evaluate(x_test, y_test)
print('loss:', loss)

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2 스코어:', r2)



print ('====================== 2. load_model 출력 ========================')
model2 = load_model('./_save/keras27_5_save_model.h5')

loss2 = model2.evaluate(x_test, y_test)
print('loss:', loss2)

y_predict2 = model2.predict(x_test)
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict2)
print('r2 스코어:', r2)

print ('====================== 3. ModelCheckPoint 출력 ========================')
model3 = load_model('./_ModelCheckPoint/keras27_5_MCP.hdf5')

loss3 = model3.evaluate(x_test, y_test)
print('loss:', loss3)

y_predict3 = model3.predict(x_test)
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2 스코어:', r2)

'''
====================== 1. 기본출력 ========================
2/2 [==============================] - 0s 998us/step - loss: 5.5013e-04 - accuracy: 1.0000
loss: [0.000550125609152019, 1.0]
r2 스코어: 0.9999876275332477
====================== 2. load_model 출력 ========================
2/2 [==============================] - 0s 1ms/step - loss: 5.5013e-04 - accuracy: 1.0000
loss: [0.000550125609152019, 1.0]
r2 스코어: 0.9999876275332477
====================== 3. ModelCheckPoint 출력 ========================
2/2 [==============================] - 0s 1ms/step - loss: 0.0092 - accuracy: 1.0000
loss: [0.009157426655292511, 1.0]
r2 스코어: 0.9999876275332477
'''

###########################드랍아웃 적용 시#############################
'''
====================== 1. 기본출력 ========================
2/2 [==============================] - 0s 996us/step - loss: 5.4472e-04 - accuracy: 1.0000
loss: [0.0005447191651910543, 1.0]
r2 스코어: 0.9999763401974323
====================== 2. load_model 출력 ========================
2/2 [==============================] - 0s 1ms/step - loss: 5.4472e-04 - accuracy: 1.0000
loss: [0.0005447191651910543, 1.0]
r2 스코어: 0.9999763401974323
====================== 3. ModelCheckPoint 출력 ========================
2/2 [==============================] - 0s 997us/step - loss: 0.0186 - accuracy: 1.0000
loss: [0.018550576642155647, 1.0]
r2 스코어: 0.9999763401974323
'''