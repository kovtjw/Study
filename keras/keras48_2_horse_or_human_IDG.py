import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.backend import binary_crossentropy

# 1,027 / 10
# 1. 데이터

train_datagen = ImageDataGenerator(
    rescale = 1./255,              
    horizontal_flip = True,        
    vertical_flip= True,           
    width_shift_range = 0.1,       
    height_shift_range= 0.1,       
    rotation_range= 5,
    zoom_range = 1.2,              
    shear_range=0.7,
    fill_mode = 'nearest',
    validation_split=0.3          
    )                   # set validation split

train_generator = train_datagen.flow_from_directory(
    '../_data/image/horse-or-human/',
    target_size=(50,50),
    batch_size=10,
    class_mode='binary',
    subset='training') # set as training data

validation_generator = train_datagen.flow_from_directory(
    '../_data/image/horse-or-human/', # same directory as training data
    target_size=(50,50),
    batch_size=10,
    class_mode='binary',
    subset='validation') # set as validation data

print(train_generator[0][0].shape)  # 719
print(validation_generator[0][0].shape) # 308



test_datagen = ImageDataGenerator(
    rescale = 1./255
)

# np.save('./_save_npy/keras48_2_train_x.npy', arr = train_generator[0][0])
# np.save('./_save_npy/keras48_2_train_y.npy', arr = train_generator[0][1])
# np.save('./_save_npy/keras48_2_test_x.npy', arr = validation_generator[0][0])
# np.save('./_save_npy/keras48_2_test_y.npy', arr = validation_generator[0][1])

# print(train_generator[0])
# print(validation_generator[0])

# xy_train = train_datagen.flow_from_directory(         
#     '../_data/image/horse-or-human/training_set',
#     target_size = (50,50),                         
#     batch_size = 10,
#     class_mode = 'binary',
#     shuffle = True,
#     )           

# xy_test = test_datagen.flow_from_directory(
#     '../_data/image/cat_dog/test_set',
#     target_size = (50,50),
#     batch_size = 10, 
#     class_mode = 'binary',
# )

# print(xy_train[0][0].shape, xy_train[0][1].shape)  # (10, 50, 50, 3) (10,)

np.save('./_save_npy/keras48_2_train_x.npy', arr = train_generator[0][0])
np.save('./_save_npy/keras48_2_train_y.npy', arr = train_generator[0][1])
np.save('./_save_npy/keras48_2_test_x.npy', arr = validation_generator[0][0])
np.save('./_save_npy/keras48_2_test_y.npy', arr = validation_generator[0][1])


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
model.compile(loss='binary_crossentropy', optimizer= 'adam', metrics=['acc'])

hist = model.fit_generator(train_generator, epochs = 20, steps_per_epoch = 72, 
                    validation_data = validation_generator,
                    validation_steps = 4,)

acc = hist.history['acc']
val_acc = hist.history['val_acc']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

print('loss:', loss[-1])
print('val_loss:', val_loss[-1])
print('acc:', acc[-1])
print('val_acc:',val_acc [-1])


# model.fit_generator(
#     train_generator,
#     steps_per_epoch = train_generator.samples // batch_size,
#     validation_data = validation_generator, 
#     validation_steps = validation_generator.samples // batch_size,
#     epochs = nb_epochs)

'''
loss: 3.5137249478666004e-20
val_loss: 0.0
acc: 1.0
val_acc: 1.0
'''