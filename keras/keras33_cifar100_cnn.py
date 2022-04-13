from tensorflow.keras.datasets import cifar100
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Conv2D, Flatten ,MaxPooling2D
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import numpy as np

(x_train, y_train), (x_test, y_test) = cifar100.load_data()
print(x_train.shape) #(50000, 3072)
print(x_test.shape) # (10000, 3072)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
scaler = StandardScaler()

n = x_train.shape[0]# 이미지갯수 50000
x_train_reshape = x_train.reshape(n,-1) #----> (50000,32,32,3) --> (50000, 32*32*3 ) 0~255
x_train_transe = scaler.fit_transform(x_train_reshape) #0~255 -> 0~1
x_train = x_train_transe.reshape(x_train.shape) #--->(50000,32,32,3) 0~1

m = x_test.shape[0]
x_test = scaler.transform(x_test.reshape(m,-1)).reshape(x_test.shape)
# print(x_train.shape,x_test.shape)  # (50000, 32, 32, 3) (10000, 32, 32, 3)

# scaler =  MaxAbsScaler()         
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)


model = Sequential() 
model.add(Conv2D(32, kernel_size = (4,4), strides = 1,
                 padding = 'same',input_shape = (32,32,3))) 
model.add(MaxPooling2D())
model.add(Conv2D(16, (3,3), activation='relu'))                     
model.add(Dropout(0.5))
model.add(Conv2D(8, (3,3), activation='relu'))
model.add(MaxPooling2D())         
model.add(Flatten()) 
model.add(Dense(400, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(300, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(100, activation = 'softmax'))



#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', patience=5, mode='auto',
                   verbose=1, restore_best_weights=False) # restore_best_weights=True : 최종값 이전에 가장 좋았던 값 도출함
mcp = ModelCheckpoint (monitor = 'val_loss', mode = 'min', verbose = 1, save_best_only=True,
                       filepath = './_ModelCheckPoint/keras27_5_MCP.hdf5')
model.fit(x_train, y_train, epochs=100, batch_size=16,
          validation_split=0.111, callbacks=[es,mcp])

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
스케일러는 전부다 2차원으로 받아들인다. >> 4차원은 불가함 >>> 2차원으로 변경해야 함 >>>> 2차원 변경 시 순서와 값은 바뀌지 않는다.

'''