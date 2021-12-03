import numpy as np

#1. 데이터
x = np.array([range(100),range(301,401),range(1,101)])
y = np.array([range(701,801)])
x = np.transpose(x)
y = np.transpose(y)
# print(x.shape,y.shape) #(100, 3) (100, 1)


#2. 모델구성
from tensorflow.keras.models import Sequential, Model # 함수형 모델
from tensorflow.keras.layers import Dense, Input


input1 = Input(shape=(3,))
dense1 = Dense(10)(input1)
dense2 = Dense(9,activation='relu')(dense1)
dense3 = Dense(8)(dense2)
output1 = Dense(1)(dense3)
model = Model(inputs=input1, outputs=output1)
model.summary()


# model = Sequential() 
# # model.add(Dense(10, input_dim=3))    
# model.add(Dense(10, input_shape=(3,)))    
# model.add(Dense(9)) 
# model.add(Dense(8))
# model.add(Dense(7))
# model.add(Dense(1))  
# model.summary()


# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# dense (Dense)                (None, 10)                40
# _________________________________________________________________
# dense_1 (Dense)              (None, 9)                 99
# _________________________________________________________________
# dense_2 (Dense)              (None, 8)                 80
# _________________________________________________________________
# dense_3 (Dense)              (None, 7)                 63        
# _________________________________________________________________
# dense_4 (Dense)              (None, 1)                 8
# =================================================================
# Total params: 290
# Trainable params: 290
# Non-trainable params: 0

# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# input_1 (InputLayer)         [(None, 3)]               0
# _________________________________________________________________
# dense (Dense)                (None, 10)                40
# _________________________________________________________________
# dense_1 (Dense)              (None, 9)                 99
# _________________________________________________________________
# dense_2 (Dense)              (None, 8)                 80
# _________________________________________________________________
# dense_3 (Dense)              (None, 1)                 9
# =================================================================
# Total params: 228
# Trainable params: 228
# Non-trainable params: 0

'''
함수형은 레이어 단을 먼저 쌓고, 모델 정의를해준다. <> sequential 
'''