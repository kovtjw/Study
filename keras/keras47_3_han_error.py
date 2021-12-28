import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.backend import binary_crossentropy

#1. DATA
train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=1.2,
    shear_range=0.7,
    fill_mode='nearest',
)

test_datagen = ImageDataGenerator(
    rescale=1./255
) #평가는 증폭해서는 안된다 


#이미지 폴더 정의 # D:\_data\image\brain
xy_train = train_datagen.flow_from_directory(
    '../_data/image/brain/train/',
    target_size=(10, 10), #사이즈는 지정된대로 바꿀수있다
    batch_size=1,
    class_mode='binary',
    shuffle=True,) #Found 160 images belonging to 2 classes.

xy_test = test_datagen.flow_from_directory(
    '../_data/image/brain/test/',
    target_size=(10, 10),
    batch_size=1, 
    class_mode='binary',) 
    

print(xy_train) 
print(xy_train[0]) 
print(xy_train[31]) 


print(xy_train[0][0].shape) #(5, 100, 100, 3) #디폴트 채널은 3, 컬러다
print(xy_train[0][1].shape) #(5,)
print(type(xy_train))       #<class 'tensorflow.python.keras.preprocessing.image.DirectoryIterator'>
print(type(xy_train[0]))    #<class 'tuple'>
print(type(xy_train[0][0])) #<class 'numpy.ndarray'>


#2. MODEL
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten

model = Sequential()
model.add(Conv2D(32, (2,2), input_shape=(150,150,3)))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))

#3. Compile, Train
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

# model.fit(xy_train[0][0], xy_train[0][1])
hist = model.fit_generator(xy_train, epochs=30, steps_per_epoch=32, #전체데이터/batch = 160/5 =32
                    validation_data=xy_test,
                    validation_steps=4,
                    )
acc = hist.history['acc']
val_acc = hist.history['val_loss']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

#그래프 그려보세요

print('loss : ', loss[-1])
print('val_loss : ', val_loss[-1])
print('acc : ', acc[-1])
print('val_acc : ', val_acc[-1])

