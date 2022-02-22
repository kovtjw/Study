import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.backend import binary_crossentropy
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D

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
    fill_mode = 'nearest'          
    )                      

test_datagen = ImageDataGenerator(
    rescale = 1./255
)

xy_train = train_datagen.flow_from_directory(         
    '../_data/image/cat_dog/training_set/training_set',
    target_size = (100,100),                         
    batch_size = 10,
    class_mode = 'categorical',
    shuffle = True,
    )           


xy_test = test_datagen.flow_from_directory(
    '../_data/image/cat_dog/test_set/test_set',
    target_size = (100,100),
    batch_size = 10, 
    class_mode = 'categorical',
)


# 2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout
from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import GlobalAvgPool2D,MaxPool2D
Xception = Xception(weights = 'imagenet', include_top = False,
              input_shape = (100, 100, 3))

model = Sequential()
model.add(Xception)
model.add(Conv2D(32,(2,2), input_shape = (100,100,3)))
model.add(GlobalAvgPool2D())
model.add(Dense(64,activation='relu'))
model.add(Dense(32,activation='relu')) 
model.add(Dense(16,activation='relu'))
model.add(Dense(2,activation='softmax'))

# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer= 'adam', metrics=['acc'])

hist = model.fit_generator(xy_train, epochs = 200, steps_per_epoch = 801, 
                    validation_data = xy_test,
                    validation_steps = 4,)


model.save('./_save/keras48_1_save.h5')

acc = hist.history['acc']
val_acc = hist.history['val_acc']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

print('loss:', loss[-1])
print('val_loss:', val_loss[-1])
print('acc:', acc[-1])
print('val_acc:',val_acc [-1])

# model.evaluate_generator

#############prtdict##############

import numpy as np
import pandas as pd

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

pic_path = './_data/image/predict/dog.PNG'
model_path = './_save/keras48_1_save.h5'

def load_my_image(img_path,show=False):
    img = image.load_img(img_path, target_size=(100,100))
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis = 0)
    img_tensor /=255.
    
    if show:
        plt.imshow(img_tensor[0])    
        plt.append('off')
        plt.show()
    
    return img_tensor

if __name__ == '__main__':
    model = load_model(model_path)
    new_img = load_my_image(pic_path)
    pred = model.predict(new_img)
    cat = pred[0][0]*100
    dog = pred[0][1]*100
    if cat > dog:
        print(f"당신은 {round(cat,2)} % 확률로 고양이 입니다")
    else:
        print(f"당신은 {round(dog,2)} % 확률로 개 입니다")

'''
loss: 0.5420612692832947
val_loss: 0.4397617280483246
acc: 0.7247970104217529
val_acc: 0.7250000238418579
'''
