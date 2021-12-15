from tensorflow.keras.utils import to_categorical
import numpy as np
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#2. 모델구성
model = Sequential()
model.add(LSTM(32,activation='relu',input_shape = (28,28)))  
model.add(Dense(128, activation='relu'))      
model.add(Dense(64, activation='relu'))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(10, activation = 'softmax'))


#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])
 
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', patience=5, mode='auto',
                   verbose=1, restore_best_weights=False) # restore_best_weights=True : 최종값 이전에 가장 좋았던 값 도출함
mcp = ModelCheckpoint (monitor = 'val_loss', mode = 'min', verbose = 1, save_best_only=True,
                       filepath = './_ModelCheckPoint/keras27_5_MCP.hdf5')
model.fit(x_train, y_train, epochs=16, batch_size=16,
          validation_split=0.3, callbacks=[es,mcp])

model.save('./_save/keras30_2_save_model.h5')


#4. 평가, 예측
print ('====================== 1. 기본출력 ========================')
loss = model.evaluate(x_test, y_test)
print('loss:', loss)

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2 스코어:', r2)