import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.backend import binary_crossentropy
from tensorflow.keras.applications import VGG19

# train_datagen = ImageDataGenerator(  
#     rescale=1./255, 
#     horizontal_flip = True,
#     #vertical_flip = True,
#     width_shift_range= 0.1,
#     height_shift_range= 0.1,
#     #rotation_range= 5,
#     zoom_range = 0.1,
#     #shear_range = 0.7,
#     validation_split=0.3,
#     fill_mode= 'nearest')                   # set validation split

# train_generator = train_datagen.flow_from_directory(
#     '../_data/image/horse-or-human/horse-or-human/',
#     target_size=(50,50),
#     batch_size=5,
#     class_mode='binary',
#     subset='training') # set as training data

# validation_generator = train_datagen.flow_from_directory(
#     '../_data/image/horse-or-human/horse-or-human/', # same directory as training data
#     target_size=(50,50),
#     batch_size=5,
#     class_mode='binary',
#     subset='validation')


x_train = np.load('./_save_npy/keras48_2_train_x.npy')
y_train = np.load('./_save_npy/keras48_2_train_y.npy')
x_test = np.load('./_save_npy/keras48_2_test_x.npy')
y_test = np.load('./_save_npy/keras48_2_test_y.npy')

# 2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout

VGG19 = VGG19(weights = 'imagenet', include_top = False,
              input_shape = (50, 50, 3))

model = Sequential()
model.add(VGG19)
# model.add(Conv2D(32,(2,2), input_shape = (50,50,3)))
model.add(Flatten())
model.add(Dense(64,activation='relu'))
model.add(Dense(32,activation='relu')) 
model.add(Dense(16,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

# 3. 컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
model.compile(loss='binary_crossentropy', optimizer= 'adam', metrics=['acc'])
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