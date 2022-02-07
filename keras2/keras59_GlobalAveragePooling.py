from tensorflow.keras.datasets import mnist 
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Dropout, Conv2D, Flatten

#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(60000,28,28,1) 
x_test = x_test.reshape(10000, 28,28,1)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#2. 모델구성
from tensorflow.keras.layers import GlobalAvgPool2D,MaxPool2D

model = Sequential() 
model.add(Conv2D(7, kernel_size = (3,3), input_shape = (28,28,1))) 
model.add(Conv2D(5, (3,3), activation='relu'))
model.add(MaxPool2D())
model.add(Dropout(0.2))
model.add(Conv2D(4, (2,2), activation='relu')) 
model.add(GlobalAvgPool2D())   # 정리하기 
# model.add(Flatten()) 
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(16, activation='relu'))
model.add(Dense(10, activation = 'softmax'))

#3. 컴파일, 훈련
import time
from tensorflow.keras.optimizers import Adam
lr = 0.0001
optimizer = Adam(learning_rate=lr)
model.compile(loss='categorical_crossentropy', optimizer = optimizer, metrics=['accuracy'])
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', patience=5, mode='auto',
                   verbose=1, restore_best_weights=False) # restore_best_weights=True : 최종값 이전에 가장 좋았던 값 도출함
mcp = ModelCheckpoint (monitor = 'val_loss', mode = 'min', verbose = 1, save_best_only=True,
                       filepath = './_ModelCheckPoint/keras27_5_MCP.hdf5')
start = time.time()
model.fit(x_train, y_train, epochs=10, batch_size=16,
          validation_split=0.3, callbacks=[es,mcp])
end = time.time()


#4. 평가, 예측
print ('====================== 1. 기본출력 ========================')
loss = model.evaluate(x_test, y_test)
print('loss:', loss)

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('learning_rate :', round(lr,4))
print('r2 스코어:', round(r2,4))
print('걸린시간 :', round(end - start,4))

