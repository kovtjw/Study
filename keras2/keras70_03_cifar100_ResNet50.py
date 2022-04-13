# 1. vgg trainabel : True, False
# 2. Flatten / GAP
# 3. Time : ... // loss : ... // acc_score : ...

from tensorflow.keras.datasets import cifar100
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions

#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar100.load_data()
x_train = x_train.reshape(50000,32,32,3)/255 
x_test = x_test.reshape(10000, 32,32,3)/255
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = preprocess_input(x_train)
x_test = preprocess_input(x_test)

#2. 모델구성
from tensorflow.keras.layers import GlobalAvgPool2D,MaxPool2D
ResNet50 = ResNet50(weights = 'imagenet', include_top = False,
              input_shape = (32, 32, 3))

model = Sequential()
model.add(ResNet50)
model.add(GlobalAvgPool2D()) 
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(16, activation='relu'))
model.add(Dense(100, activation = 'softmax'))

#3. 컴파일, 훈련
import time
from tensorflow.keras.optimizers import Adam
##############################################
learning_rate = 0.0001
optimizer = Adam(learning_rate=learning_rate)
############################################################################################
model.compile(loss='categorical_crossentropy', optimizer = optimizer, metrics=['accuracy'])
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
############################################################################################
es = EarlyStopping(monitor='val_loss', patience=15, mode='auto',
                   verbose=1, restore_best_weights=True)
ReduceLR = ReduceLROnPlateau(monitor='val_loss', patience=5, mode = 'min',
                             verbose=3, factor=0.5)
# factor = 0.5 : patience의 기간 동안 loss가 감소되지 않으면(갱신되지 않으면), 
# 50프로의 러닝레이트를 감소시키겠다.
############################################################################################

start = time.time()
model.fit(x_train, y_train, epochs=100, batch_size=16,
          validation_split=0.25, callbacks=[es, ReduceLR])
end = time.time()

#4. 평가, 예측
print ('====================== 1. 기본출력 ========================')
loss = model.evaluate(x_test, y_test)
print('loss:', loss[0])
print('걸린시간 :', (end - start))
print('acc:', loss[1])

'''
====================== 1. 기본출력 ========================
313/313 [==============================] - 5s 17ms/step - loss: 2.3734 - accuracy: 0.4842
loss: 2.3734469413757324
걸린시간 : 1416.972823381424
acc: 0.48420000076293945
val_loss
'''
'''
스케일링 후
loss: 4.604944705963135
걸린시간 : 617.707542181015
acc: 0.00989999994635582
'''