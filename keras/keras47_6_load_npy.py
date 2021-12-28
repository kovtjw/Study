<<<<<<< HEAD
import numpy as np

# np.save('./_save_npy/keras47_5_train_x.npy', arr = xy_train[0][0])
# np.save('./_save_npy/keras47_5_train_y.npy', arr = xy_train[0][1])
# np.save('./_save_npy/keras47_5_test_x.npy', arr = xy_test[0][0])
# np.save('./_save_npy/keras47_5_test_y.npy', arr = xy_test[0][1])

x_train = np.load('./_save_npy/keras47_5_train_x.npy')
y_train = np.load('./_save_npy/keras47_5_train_y.npy')
x_test = np.load('./_save_npy/keras47_5_test_x.npy')
y_test = np.load('./_save_npy/keras47_5_test_y.npy')



# 2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout

model = Sequential()
model.add(Conv2D(32,(2,2), input_shape = (150,150,3)))
model.add(Flatten())
model.add(Dense(64,activation='relu'))
model.add(Dense(32,activation='relu')) 
model.add(Dense(16,activation='relu'))
model.add(Dense(1,activation='sigmoid'))


# 3. 컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
model.compile(loss='binary_crossentropy', optimizer= 'adam', metrics=['acc'])
es = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1) 
hist = model.fit(x_train, y_train, epochs = 50, batch_size = 5,
                 validation_split= 0.2,callbacks=[es])  

acc = hist.history['acc']
val_acc = hist.history['val_acc']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

# 점심 때 그래프 그려 보기
print('loss:', loss[-1])
print('val_loss:', val_loss[-1])
print('acc:', acc[-1])
print('val_acc:',val_acc [-1])

'''
loss: 0.00016787357162684202
val_loss: 0.013173221610486507
acc: 1.0
val_acc: 1.0
=======
import numpy as np

# np.save('./_save_npy/keras47_5_train_x.npy', arr = xy_train[0][0])
# np.save('./_save_npy/keras47_5_train_y.npy', arr = xy_train[0][1])
# np.save('./_save_npy/keras47_5_test_x.npy', arr = xy_test[0][0])
# np.save('./_save_npy/keras47_5_test_y.npy', arr = xy_test[0][1])

x_train = np.load('./_save_npy/keras47_5_train_x.npy')
y_train = np.load('./_save_npy/keras47_5_train_y.npy')
x_test = np.load('./_save_npy/keras47_5_test_x.npy')
y_test = np.load('./_save_npy/keras47_5_test_y.npy')



# 2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout

model = Sequential()
model.add(Conv2D(32,(2,2), input_shape = (150,150,3)))
model.add(Flatten())
model.add(Dense(64,activation='relu'))
model.add(Dense(32,activation='relu')) 
model.add(Dense(16,activation='relu'))
model.add(Dense(1,activation='sigmoid'))


# 3. 컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
model.compile(loss='binary_crossentropy', optimizer= 'adam', metrics=['acc'])
es = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1) 
hist = model.fit(x_train, y_train, epochs = 50, batch_size = 5,
                 validation_split= 0.2,callbacks=[es])  

acc = hist.history['acc']
val_acc = hist.history['val_acc']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

# 점심 때 그래프 그려 보기
print('loss:', loss[-1])
print('val_loss:', val_loss[-1])
print('acc:', acc[-1])
print('val_acc:',val_acc [-1])

'''
loss: 0.00016787357162684202
val_loss: 0.013173221610486507
acc: 1.0
val_acc: 1.0
>>>>>>> ebd7c4e88163fa934a01d2f951f2098401e8135f
'''