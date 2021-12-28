import numpy as np
import pandas as pd

x_train = np.load('./_save_npy/keras48_1_train_x.npy')
y_train = np.load('./_save_npy/keras48_1_train_y.npy')
x_test = np.load('./_save_npy/keras48_1_test_x.npy')
y_test = np.load('./_save_npy/keras48_1_test_y.npy')


# 2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout

model = Sequential()
model.add(Conv2D(32,(2,2), input_shape = (50,50,3)))
model.add(Flatten())
model.add(Dense(64,activation='relu'))
model.add(Dense(32,activation='relu')) 
model.add(Dense(16,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

# 3. 컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
model.compile(loss='binary_crossentropy', optimizer= 'adam', metrics=['acc'])
es = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1) 

mcp = ModelCheckpoint(monitor= 'val_loss', mode = 'min', verbose =1, save_best_only=True,
                      filepath = './_save/keras48_1_save_weights21.h5')
hist = model.fit(x_train, y_train, epochs = 20, batch_size = 32,
                 validation_split= 0.2,callbacks=[es,mcp])  
model.save_weights('./_save/keras48_1_save_weights1111.h5')


acc = hist.history['acc']
val_acc = hist.history['val_acc']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

print('loss:', loss[-1])
print('val_loss:', val_loss[-1])
print('acc:', acc[-1])
print('val_acc:',val_acc [-1])


'''
loss: 2.1166661312577162e-08
val_loss: 1.9264185732725814e-11
acc: 1.0
val_acc: 1.0
'''