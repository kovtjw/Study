from tensorflow.keras.datasets import cifar100
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
import numpy as np
from tensorflow.keras.utils import to_categorical
(x_train, y_train), (x_test, y_test) = cifar100.load_data()
# print(x_train.shape) 
# print(x_test.shape)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

scaler = StandardScaler()

n = x_train.shape[0]# 이미지갯수 50000
x_train_reshape = x_train.reshape(n,-1) 
x_train = scaler.fit_transform(x_train_reshape) 
# x_train = x_train_transe.reshape(x_train.shape) 

m = x_test.shape[0]
x_test = scaler.transform(x_test.reshape(m,-1))
#2. 모델구성
model = Sequential()
model.add(Dense(100, input_shape=(3072,)))
# model.add(Dense(64, input_shape=(784, )))  # 위와 동일함
model.add(Dense(80, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(130, activation='sigmoid'))
model.add(Dense(80, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(5))
model.add(Dense(100, activation = 'softmax'))

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])
 
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', patience=5, mode='auto',
                   verbose=1, restore_best_weights=False) # restore_best_weights=True : 최종값 이전에 가장 좋았던 값 도출함
mcp = ModelCheckpoint (monitor = 'val_loss', mode = 'min', verbose = 1, save_best_only=True,
                       filepath = './_ModelCheckPoint/keras27_5_MCP.hdf5')
model.fit(x_train, y_train, epochs=16, batch_size=16,
          validation_split=0.1, callbacks=[es,mcp])

model.save('./_save/keras30_2_save_model.h5')

#4. 평가, 예측
print ('====================== 1. 기본출력 ========================')
loss = model.evaluate(x_test, y_test)
print('loss:', loss)

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2 스코어:', r2)

'''
====================== 1. 기본출력 ========================
313/313 [==============================] - 0s 786us/step - loss: 3.9295 - accuracy: 0.0960
loss: [3.9294745922088623, 0.09600000083446503]
'''