import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPool2D, Dropout
import time
import warnings
warnings.filterwarnings('ignore')

train_datagen = ImageDataGenerator(  
    rescale=1./255, 
    horizontal_flip = True,
    #vertical_flip = True,
    width_shift_range= 0.1,
    height_shift_range= 0.1,
    #rotation_range= 5,
    zoom_range = 0.1,
    #shear_range = 0.7,
    fill_mode= 'nearest')

test_datagen = ImageDataGenerator(
    rescale = 1./255
)

xy_train = train_datagen.flow_from_directory(        # 160 
    '../_data/image/brain/train',
    target_size = (150,150),                         
    batch_size = 160,
    class_mode = 'binary',
    shuffle = True,
    )

xy_test = test_datagen.flow_from_directory(         # 120
    '../_data/image/brain/test',
    target_size = (150,150),
    batch_size = 300, 
    class_mode = 'binary',
)


print(xy_train)
# # 증폭 데이터 생성
augment_size = 5000
randidx = np.random.randint(160, size = augment_size)

x_augmented = xy_train[0][0][randidx].copy()     # x값
y_augmented = xy_train[0][1][randidx].copy()     # y값

# print(xy_train[0][0].shape)

x_train = xy_train[0][0].reshape(xy_train[0][0].shape[0],150,150,3)
x_test = xy_test[0][0].reshape(xy_test[0][0].shape[0],150,150,3)

augmented_data = train_datagen.flow(x_augmented,y_augmented,batch_size = 28, shuffle = False)

x = np.concatenate((x_train, augmented_data[0][0]))  
y = np.concatenate((xy_train[0][1], augmented_data[0][1]))

# print(x.shape)    # (188, 150, 150, 3)

model = Sequential()
model.add(Conv2D(64,(2,2), input_shape = (150,150,3)))
model.add(Conv2D(64,(2,2)))
model.add(Flatten())
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(1,activation='sigmoid'))

# #3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])  

start = time.time()
model.fit(x, y, epochs=10, steps_per_epoch=10) # (100000/32)
end = time.time() - start
print("걸린 시간 : ", round(end, 2))

#4. 예측 
loss = model.evaluate(xy_test)
print("loss : ", loss)


# loss: 0.1224 - acc: 0.8830