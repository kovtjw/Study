import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.backend import binary_crossentropy

x_train = np.load('../_data/_save_npy/keras48_3_train_x.npy')
y_train = np.load('../_data/_save_npy/keras48_3_train_y.npy')
x_test = np.load('../_data/_save_npy/keras48_3_test_x.npy')
y_test = np.load('../_data/_save_npy/keras48_3_test_y.npy')

# 2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout

model = Sequential()
model.add(Conv2D(32,(2,2), input_shape = (50,50,3)))
model.add(Flatten())
model.add(Dense(64,activation='relu'))
model.add(Dense(32,activation='relu')) 
model.add(Dense(16,activation='relu'))
model.add(Dense(3,activation='softmax'))

# 3. 컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
model.compile(loss='categorical_crossentropy', optimizer= 'adam', metrics=['acc'])
es = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1) 
hist = model.fit(x_train, y_train, epochs = 20, batch_size = 32,
                 validation_split= 0.2,callbacks=[es])  

acc = hist.history['acc']
val_acc = hist.history['val_acc']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

print('loss:', loss[-1])
print('val_loss:', val_loss[-1])
print('acc:', acc[-1])
print('val_acc:',val_acc [-1])