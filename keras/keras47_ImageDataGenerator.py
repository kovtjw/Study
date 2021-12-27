import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 클래스에 대한 정의
train_datagen = ImageDataGenerator(
    rescale = 1./255,              # minmax scaler 
    horizontal_flip = True,        # 수평으로 반전 시키겠다.
    vertical_flip= True,           # 수직으로 반전 시키겠다.
    width_shift_range = 0.1,       # 10% 만큼 양옆으로 이동시킨다.
    height_shift_range= 0.1,       # 10% 만큼 위, 아래로 이동시킨다.
    rotation_range= 5,
    zoom_range = 1.2,              # 확대시킨다.
    shear_range=0.7,
    fill_mode = 'nearest'          
    )                      

test_datagen = ImageDataGenerator(
    rescale = 1./255
)
# 왜 test data gen은 스케일러만 적용했을까??? 평가 데이터는 원래의 이미지를 사용해야 하기 때문에 
# :\_data\image\brain

xy_train = train_datagen.flow_from_directory(        # directory = folder  
    '../_data/image/brain/train',
    target_size = (150,150),                         # 원하는 사이즈로 지정할 수 있다. > 원본과 비슷한 사이즈로 지정해야 함
    batch_size = 5,
    class_mode = 'binary',
    shuffle = True,
    )           

xy_test = test_datagen.flow_from_directory(
    '../_data/image/brain/test',
    target_size = (150,150),
    batch_size = 5, 
    class_mode = 'binary',
)