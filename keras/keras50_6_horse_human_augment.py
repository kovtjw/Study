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
    validation_split=0.3,
    fill_mode= 'nearest')


train_generator = train_datagen.flow_from_directory(
    '../_data/image/horse-or-human/horse-or-human/',
    target_size=(150,150),
    batch_size=1027,
    class_mode='binary',
    subset='training') # set as training data

validation_generator = train_datagen.flow_from_directory(
    '../_data/image/horse-or-human/horse-or-human/', # same directory as training data
    target_size=(150,150),
    batch_size=308,
    class_mode='binary',
    subset='validation')

print(train_generator[0][0].shape)  # (719, 150, 150, 3)
print(validation_generator[0][0].shape) # (308, 150, 150, 3)

# # 증폭 데이터 생성
augment_size = 1000
randidx = np.random.randint(1027, size = augment_size)

x_augmented = train_generator[0][0][randidx].copy()     # x값
y_augmented = train_generator[0][1][randidx].copy() 

x_train = train_generator[0][0].reshape(train_generator[0][0].shape[0],150,150,3)
x_test = validation_generator[0][0].reshape(validation_generator[0][0].shape[0],150,150,3)

augmented_data = train_datagen.flow(x_augmented,y_augmented,batch_size = 1000, shuffle = False)

x = np.concatenate((x_train, augmented_data[0][0]))  
y = np.concatenate((train_generator[0][1], augmented_data[0][1]))

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
model.fit(x, y, epochs=10, steps_per_epoch=1000) 
end = time.time() - start
print("걸린 시간 : ", round(end, 2))

#4. 예측 
loss = model.evaluate(validation_generator)
print("loss : ", loss)
