import tensorflow as tf
from tensorflow.keras.layers import Conv2D

import matplotlib.pyplot as plt
plt.style.use('seaborn-white')

import numpy as np
from sklearn.datasets import load_sample_image
# china = load_sample_image('china.jpg') / 255
# print(china.dtype)
# print(china.shape)

# plt.imshow(china)
# plt.show()

# flower = load_sample_image('flower.jpg') / 255
# print(flower.dtype)
# print(flower.shape)

# images = np.array([china, flower])
# batch_size, height, width, channels = images.shape
# print(images.shape)

# filters = np.zeros(shape=(7,7, channels, 2), dtype = np.float32)
# filters[:,3,:,0] = 1
# filters[3,:,:,1] = 1

# print(filters.shape)

# outputs = tf.nn.conv2d(images, filters, strides =1, padding = 'SAME')
# print(outputs.shape)
# plt.imshow(outputs[0,:,:,1], cmap = 'gray')
# plt.show()

# conv = Conv2D(filters=32, kernel_size=3, strides=1,
#               padding = 'SAME', activation='relu')

import tensorflow as tf
from tensorflow.keras.layers import MaxPool2D, Lambda

# output = tf.nn.max_pool(images,
#                         ksize=(1,1,1,3),
#                         strides=(1,1,1,3),
#                         padding='VALID')

# output_keras = Lambda(
#     lambda X : tf.nn.max_pool(X, ksize=(1,1,1,3), strides=(1,1,1,3), padding='VALID')
# )

# max_pool = MaxPool2D(pool_size=2)

# flower = load_sample_image('flower.jpg') / 255
# print(flower.dtype)
# print(flower.shape)

# flower = np.expand_dims(flower, axis=0)  # 디멘션 추가
# flower.shape

# output = Conv2D(filters=32, kernel_size=3, strides=1, padding='same',activation='relu')(flower)
# output = MaxPool2D(pool_size=2)(output)

# output.shape   

# plt.imshow(output[0,:,:,8], cmap='gray')
# plt.show()

from tensorflow.keras.layers import AvgPool2D

# flower.shape

# output = Conv2D(filters=32, kernel_size=3, strides=1, padding='same',activation='relu')(flower)
# output = AvgPool2D(pool_size=2)(output)

# output.shape

# plt.imshow(output[0,:,:,2],cmap='gray')
# plt.show()

from tensorflow.keras.layers import GlobalAvgPool2D
# output = Conv2D(filters=32, kernel_size=3, strides=1, padding = 'same', activation = 'relu')(flower)
# output = GlobalAvgPool2D()(output)

# output.shape  #(427, 640, 3)


#### 예제로 보는 CNN 구조와 학습

import datetime
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')

from tensorflow.keras import Model
from tensorflow.keras. models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, AvgPool2D, Dropout

from tensorflow.keras import datasets
from tensorflow.keras.utils import to_categorical, plot_model
(x_train,y_train),(x_test,y_test) = datasets.fashion_mnist.load_data()

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

'''
(60000, 28, 28)
(60000,)
(10000, 28, 28)
(10000,)
'''

x_train = x_train[:,:,:,np.newaxis]   # 축을 추가
x_test = x_test[:,:,:,np.newaxis]
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

num_classes = 10
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)
print(y_train.shape)
print(y_test.shape)
'''
(60000, 10)
(10000, 10)
'''

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train /= 255
x_test /= 255

def build():
    model = Sequential([Conv2D(64,7,activation='relu', padding = 'same', input_shape = [28,28,1]),
                        MaxPool2D(pool_size=2),
                        Conv2D(128,3,activation='relu',padding='same'),
                        MaxPool2D(pool_size=2),
                        Conv2D(256,3,activation='relu',padding='same'),
                        MaxPool2D(2),
                        Flatten(),
                        Dense(128, activation='relu'),
                        Dropout(0.5),
                        Dense(64, activation='relu'),
                        Dropout(0.5),    
                        Dense(10, activation='softmax')])
    return model

model = build()

model.compile(optimizer = 'adam',
              loss = ' categorical_crossentropy',
              metrics=['accuracy'])    

model.summary()

callbacks = [tf.keras.callbacks.TensorBoard(log_dir='./logs')]
EPOCHS = 29
BATCH_SIZE = 200
VERBOSE = 1

hist = model.fit(x_train, y_train,
                 epochs = EPOCHS,
                 batch_size=BATCH_SIZE,
                 validation_split = 0.3,
                 callbacks = callbacks,
                 verbose = VERBOSE)

