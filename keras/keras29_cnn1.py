from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout

model = Sequential() 
model.add(Conv2D(10, kernel_size = (2,2), input_shape = (10,10,1)))  # 4,4,10
#input_shape = (a,b,c) > kernel_size = (d,e) = (a-d+1, b-e+1)  // 

# model.add(Conv2D(5, (3,3), activation='relu'))                     # 3,3,5
# model.add(Dropout(0.2))
# model.add(Conv2D(7, (2,2), activation='relu'))                     # 2,2,7
# model.add(Flatten())  # flatten (Flatten)            (None, 252)               0
# model.add(Dense(64))
# model.add(Dropout(0.2))
# model.add(Dense(16))
# model.add(Dense(5, activation = 'softmax'))

# kernel_size = (2,2) = 2,2로 잘라서 사용하겠다. >> 파라미터 튜닝에 속함 
# input_dim 은 1차원만 받을 수 있다. >> input_shape를 사용해야 함 >>> 1000 x 300 x 300 x 3 >> input_shape = (300, 300, 3)
# 데이터 수집과 정제를 할 때 모든 데이터의 shape(모양)가 동일해야한다. 
# input_shape = (5,5,1))) 마지막 1 생략 불가함 >> 무조건 1 혹은 3을 기입
# input_shape = (5,5,1))) >> (4,4,10)
# flatten (Flatten) : 4차원의 데이터를 2차원의 데이터로 변경 >> 한 줄로 쭉편다 >>> Dense로 받아준다.
model.summary()

# 유천, 창민, 재중, 지영, 규리 - > (5,) - >(5,1)
# LabelEncoder
# 0,1,2,3,4 

'''
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d (Conv2D)              (None, 4, 4, 10)          50
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 3, 3, 5)           205
=================================================================
Total params: 255
Trainable params: 255
Non-trainable params: 0

############################ 파라미터 갯수 구하는 법 ############################
model.add(Conv2D(c, kernel_size = (a,b) , input_shape = (10,10,1)))
(필터 크기 axb) x  (입력 채널(RGB)) x (출력 채널 c) + (출력 채널 c bias)
두 번째 연산은
위 레이어의 출력채널 c를 아래의 출력 채널값과 곱하여 계산한다. 
model.add(Conv2D(필터, kernel_size = (a,b) , input_shape = (10,10,채널)))

'''