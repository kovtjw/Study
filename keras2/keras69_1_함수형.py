from re import I
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.applications import VGG16
from tensorflow.keras.datasets import cifar100
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D

#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar100.load_data()
x_train = x_train.reshape(50000,32,32,3)/255 
x_test = x_test.reshape(10000, 32,32,3)/255
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

from tensorflow.keras.layers import GlobalAvgPool2D,MaxPool2D
input = Input(shape = (32, 32, 3))
vgg16 = VGG16(include_top=False)(input)
gap = GlobalAvgPool2D()(vgg16)
hidden1 = Dense(32)(gap)
dropout1 = Dropout(0.2)(hidden1)
hidden2 = Dense(16)(dropout1)
output1 = Dense(100, activation='softmax')(hidden2)
model = Model(inputs = input, outputs = output1)
model.summary()

#3. 컴파일, 훈련
import time
from tensorflow.keras.optimizers import Adam
##############################################
learning_rate = 0.00001
optimizer = Adam(learning_rate=learning_rate)
############################################################################################
model.compile(loss='categorical_crossentropy', optimizer = optimizer, metrics=['accuracy'])
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
############################################################################################
es = EarlyStopping(monitor='val_loss', patience=15, mode='auto',
                   verbose=1, restore_best_weights=True)
ReduceLR = ReduceLROnPlateau(monitor='val_loss', patience=5, mode = 'min',
                             verbose=3, factor=0.5)

start = time.time()
model.fit(x_train, y_train, epochs=3, batch_size=32,
          validation_split=0.25, callbacks=[es, ReduceLR])
end = time.time()

#4. 평가, 예측
print ('====================== 1. 기본출력 ========================')
loss = model.evaluate(x_test, y_test)
print('loss:', loss[0])
print('걸린시간 :', (end - start))
print('acc:', loss[1])