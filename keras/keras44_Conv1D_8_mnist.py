import numpy as np
from sklearn import metrics
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten
from tensorflow.keras.utils import to_categorical
import time
#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


#2. 모델구성
model = Sequential()
model.add(Conv1D(32,2,activation='relu',input_shape = (28,28)))
model.add(Flatten())   
model.add(Dense(128, activation='relu'))      
model.add(Dense(64, activation='relu'))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(10, activation = 'softmax'))

#3. 컴파일, 훈련
start = time.time()
model.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics=['accuracy','mae'])
model.fit(x_train, y_train, epochs=16, batch_size=16,
          validation_split=0.3)
end = time.time()- start


#4. 평가, 예측
print ('====================== 1. 기본출력 ========================')
loss = model.evaluate(x_test, y_test)
print('loss:', loss)

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2 스코어:', r2)
print("걸린시간 : ", round(end, 3), '초')
'''
loss: [0.14794810116291046, 0.9739000201225281]
r2 스코어: 0.9505592835516794
걸린시간 :  47.479 초
'''