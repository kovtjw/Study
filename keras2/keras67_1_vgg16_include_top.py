import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.applications import VGG16  # layer가 16개

# model = VGG16()  : 전이학습의 hello world
model = VGG16(weights = 'imagenet', include_top = False,  # >> Faslse로 하면 shape 조절이 가능하다 
              input_shape = (32, 32, 3))
model.summary()
print(len(model.weights))
print(len(model.trainable_weights))

# include_top = True 
# 1. fc layer 원래 것을 그대로 쓴다.
# 2. input_shape = (224,224,3) 으로 고정된다 >> 변경 불가
'''
___________________include_top = True____________________

 fc1 (Dense)                 (None, 4096)              102764544

 fc2 (Dense)                 (None, 4096)              16781312

 predictions (Dense)         (None, 1000)              4097000

=================================================================
Total params: 138,357,544
Trainable params: 138,357,544
Non-trainable params: 0
_________________________________________________________________
'''

# include_top = False
# 1. fc layer 원래 것을 없앤다. >> customizing이 가능하다.
# 2. input_shape 변경 가능

'''
___________________include_top = False____________________
Layer (type)                Output Shape              Param #   
=================================================================
input_1 (InputLayer)        [(None, 224, 224, 3)]     0

block1_conv1 (Conv2D)       (None, 224, 224, 64)      1792      

block1_conv2 (Conv2D)       (None, 224, 224, 64)      36928     
.....
block5_conv3 (Conv2D)       (None, 14, 14, 512)       2359808

block5_pool (MaxPooling2D)  (None, 7, 7, 512)         0

=================================================================
Total params: 14,714,688
Trainable params: 14,714,688
Non-trainable params: 0
'''
