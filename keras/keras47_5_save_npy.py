<<<<<<< HEAD
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.backend import binary_crossentropy
import matplotlib.pyplot as plt

# 1. 데이터

# 클래스에 대한 정의
train_datagen = ImageDataGenerator(
    rescale = 1./255,)              
    # horizontal_flip = True,        
    # vertical_flip= True,           
    # width_shift_range = 0.1,       
    # height_shift_range= 0.1,       
    # rotation_range= 5,
    # zoom_range = 1.2,              
    # shear_range=0.7,
    # fill_mode = 'nearest'          
    # )                      

test_datagen = ImageDataGenerator(
    rescale = 1./255
)

xy_train = train_datagen.flow_from_directory(         
    '../_data/image/brain/train',
    target_size = (150,150),                         
    batch_size = 200,
    class_mode = 'binary',
    shuffle = True,
    )           


xy_test = test_datagen.flow_from_directory(
    '../_data/image/brain/test',
    target_size = (150,150),
    batch_size = 200, 
    class_mode = 'binary',
)

# print(xy_train[0][0].shape, xy_train[0][1].shape) 

print(xy_train[0][0].shape,xy_train[0][1].shape)     # (160, 150, 150, 3)  (160,)
print(xy_test[0][0].shape,xy_test[0][1].shape)       # (120, 150, 150, 3)  (120,)

np.save('./_save_npy/keras47_5_train_x.npy', arr = xy_train[0][0])
np.save('./_save_npy/keras47_5_train_y.npy', arr = xy_train[0][1])
np.save('./_save_npy/keras47_5_test_x.npy', arr = xy_test[0][0])
np.save('./_save_npy/keras47_5_test_y.npy', arr = xy_test[0][1])



import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.backend import binary_crossentropy
import matplotlib.pyplot as plt

# 1. 데이터

# 클래스에 대한 정의
train_datagen = ImageDataGenerator(
    rescale = 1./255,)              
    # horizontal_flip = True,        
    # vertical_flip= True,           
    # width_shift_range = 0.1,       
    # height_shift_range= 0.1,       
    # rotation_range= 5,
    # zoom_range = 1.2,              
    # shear_range=0.7,
    # fill_mode = 'nearest'          
    # )                      

test_datagen = ImageDataGenerator(
    rescale = 1./255
)

xy_train = train_datagen.flow_from_directory(         
    '../_data/image/brain/train',
    target_size = (150,150),                         
    batch_size = 200,
    class_mode = 'binary',
    shuffle = True,
    )           


xy_test = test_datagen.flow_from_directory(
    '../_data/image/brain/test',
    target_size = (150,150),
    batch_size = 200, 
    class_mode = 'binary',
)

# print(xy_train[0][0].shape, xy_train[0][1].shape) 

print(xy_train[0][0].shape,xy_train[0][1].shape)     # (160, 150, 150, 3)  (160,)
print(xy_test[0][0].shape,xy_test[0][1].shape)       # (120, 150, 150, 3)  (120,)

np.save('./_save_npy/keras47_5_train_x.npy', arr = xy_train[0][0])
np.save('./_save_npy/keras47_5_train_y.npy', arr = xy_train[0][1])
np.save('./_save_npy/keras47_5_test_x.npy', arr = xy_test[0][0])
np.save('./_save_npy/keras47_5_test_y.npy', arr = xy_test[0][1])




=======
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.backend import binary_crossentropy
import matplotlib.pyplot as plt

# 1. 데이터

# 클래스에 대한 정의
train_datagen = ImageDataGenerator(
    rescale = 1./255,)              
    # horizontal_flip = True,        
    # vertical_flip= True,           
    # width_shift_range = 0.1,       
    # height_shift_range= 0.1,       
    # rotation_range= 5,
    # zoom_range = 1.2,              
    # shear_range=0.7,
    # fill_mode = 'nearest'          
    # )                      

test_datagen = ImageDataGenerator(
    rescale = 1./255
)

xy_train = train_datagen.flow_from_directory(         
    '../_data/image/brain/train',
    target_size = (150,150),                         
    batch_size = 200,
    class_mode = 'binary',
    shuffle = True,
    )           


xy_test = test_datagen.flow_from_directory(
    '../_data/image/brain/test',
    target_size = (150,150),
    batch_size = 200, 
    class_mode = 'binary',
)

# print(xy_train[0][0].shape, xy_train[0][1].shape) 

print(xy_train[0][0].shape,xy_train[0][1].shape)     # (160, 150, 150, 3)  (160,)
print(xy_test[0][0].shape,xy_test[0][1].shape)       # (120, 150, 150, 3)  (120,)

np.save('./_save_npy/keras47_5_train_x.npy', arr = xy_train[0][0])
np.save('./_save_npy/keras47_5_train_y.npy', arr = xy_train[0][1])
np.save('./_save_npy/keras47_5_test_x.npy', arr = xy_test[0][0])
np.save('./_save_npy/keras47_5_test_y.npy', arr = xy_test[0][1])

