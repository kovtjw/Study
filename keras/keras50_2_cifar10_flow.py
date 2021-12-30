from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Conv2D, Flatten ,MaxPooling2D
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
from tensorflow.keras.utils import to_categorical
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator



(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)
# print(x_train.shape,y_train.shape)   # (50000, 32, 32, 3) (50000, 1)
# print(x_test.shape,y_test.shape)     # (10000, 32, 32, 3) (10000, 1)


train_datagen = ImageDataGenerator(
    rescale = 1./255,              
    horizontal_flip = True,        
    vertical_flip= False,           
    width_shift_range = 0.1,       
    height_shift_range= 0.1,       
    # rotation_range= 5,
    zoom_range = 0.1,              
    # shear_range=0.7,
    fill_mode = 'nearest'          
    )


augment_size = 50000
randidx =  np.random.randint(x_train.shape[0], size = augment_size)  

x_augmented = x_train[randidx].copy()
y_augmented = y_train[randidx].copy()

# print(x_augmented.shape[0].shape)

x_augmented = x_augmented.reshape(50000,32,32,3)

x_train = x_train.reshape(50000,32,32,3)
x_test = x_test.reshape(x_test.shape[0],32,32,3)

xy_train = train_datagen.flow(x_augmented, y_augmented,
                                 batch_size=augment_size, 
                                 shuffle=False)

x = np.concatenate((x_train, xy_train[0][0]))  # (100000, 28, 28, 1)
y = np.concatenate((y_train, xy_train[0][1]))

# x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],x_test.shape[2],1)

# print(x.shape,y.shape)

model = Sequential() 
model.add(Conv2D(7, kernel_size = (2,2), strides = 1,
                 padding = 'same',input_shape = (32,32,3))) 
model.add(MaxPooling2D())
model.add(Conv2D(5, (3,3), activation='relu'))                     
model.add(Dropout(0.2))
model.add(Conv2D(4, (2,2), activation='relu'))         
model.add(Flatten()) 
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(16, activation='relu'))
model.add(Dense(10, activation = 'softmax'))


#3. 컴파일, 훈련
model.compile(loss='sparse_categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])

model.fit(x, y, epochs=10, steps_per_epoch=1000)


#4. 평가, 예측
print ('====================== 1. 기본출력 ========================')
loss = model.evaluate(x_test, y_test)
print('loss:', loss)

y_predict = model.predict_generator(x_test)


y_predict=np.argmax(y_predict, axis=1)

from sklearn.metrics import accuracy_score   # accuracy_score 분류에서 사용
a = accuracy_score(y_test, y_predict)
print('acc score:', a)
