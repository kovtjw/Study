from tensorflow.keras.datasets import mnist, cifar10
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Dropout, Conv2D, Flatten

#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.reshape(50000,32,32,3)/255 
x_test = x_test.reshape(10000, 32,32,3)/255
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#2. 모델구성
from tensorflow.keras.layers import GlobalAvgPool2D,MaxPool2D

model = Sequential() 
model.add(Conv2D(128, kernel_size = (3,3), input_shape = (32,32,3)))
model.add(MaxPool2D())
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPool2D())
model.add(Dropout(0.2))
model.add(Conv2D(32, (2,2), activation='relu')) 
model.add(GlobalAvgPool2D()) 
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(16, activation='relu'))
model.add(Dense(10, activation = 'softmax'))

#3. 컴파일, 훈련
import time
from tensorflow.keras.optimizers import Adam
##############################################
learning_rate = 0.001
optimizer = Adam(learning_rate=learning_rate)
############################################################################################
model.compile(loss='categorical_crossentropy', optimizer = optimizer, metrics=['accuracy'])
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
############################################################################################
es = EarlyStopping(monitor='val_loss', patience=15, mode='auto',
                   verbose=1, restore_best_weights=False)
ReduceLR = ReduceLROnPlateau(monitor='val_loss', patience=5, mode = 'min',
                             verbose=3, factor=0.5)
# factor = 0.5 : patience의 기간 동안 loss가 감소되지 않으면(갱신되지 않으면), 
# 50프로의 러닝레이트를 감소시키겠다.
############################################################################################

start = time.time()
model.fit(x_train, y_train, epochs=300, batch_size=64,
          validation_split=0.25, callbacks=[es, ReduceLR])
end = time.time()


#4. 평가, 예측
print ('====================== 1. 기본출력 ========================')
loss = model.evaluate(x_test, y_test)
print('loss:', loss)
y_predict = model.predict(x_test)
print('learning_rate :', round(ReduceLR,4))
print('걸린시간 :', round(end - start,4))