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
# Found 160 images belonging to 2 classes.

xy_test = test_datagen.flow_from_directory(
    '../_data/image/brain/test',
    target_size = (150,150),
    batch_size = 5, 
    class_mode = 'binary',
)
# Found 120 images belonging to 2 classes.

# print(xy_train)
# <tensorflow.python.keras.preprocessing.image.DirectoryIterator object at 0x000001DCB78B5F40>
# print(xy_train[31])   
#  y값 : array([1., 0., 1., 0., 0.], dtype=float32)) 5개 > batch_size를 5로 나누었기 때문에 // 160을 batch 로 나눈 값까지 나온다.
# print(xy_train[0][0]) # x 값
# print(xy_train[0][1]) # y 값
# print(xy_train[0][2]) error
print(xy_train[0][0].shape, xy_train[0][1].shape) 
#  첫 번째의 x 값의 shape(5, 150, 150, 3)    (batch, 이미지의 가로세로 데이터, 흑백or컬러) 
# y 값의 shape(5,)

print(type(xy_train))   # <class 'tensorflow.python.keras.preprocessing.image.DirectoryIterator'>
print(type(xy_train[0]))  # <class 'tuple'>
print(type(xy_train[0][0])) # x 값 <class 'numpy.ndarray'>
print(type(xy_train[0][1])) # y 값 <class 'numpy.ndarray'>