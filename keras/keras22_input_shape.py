import numpy as np

#1. 데이터
x = np.array([range(100),range(301,401),range(1,101)])
y = np.array([range(701,801)])
x = np.transpose(x)
y = np.transpose(y)
# print(x.shape,y.shape) #(100, 3) (100, 1)


#2. 모델구성
from tensorflow.keras.models import Sequential, Model # 함수형 모델
from tensorflow.keras.layers import Dense

model = Sequential() 
model.add(Dense(10, input_dim=3))    # (100,3) - > (N,3)
# model.add(Dense(10, input_shape=(3,)))   # 2차원 형식일 때 위의 x 값을 스칼라 형식으로 입력해 준다. 
model.add(Dense(9)) 
model.add(Dense(8))
model.add(Dense(7))
model.add(Dense(1))  # output은 y의 열, 컬럼, 특성의 갯수이다. 
model.summary()


'''
2차원 에서 사용하는 input_dim=3은 다차원에서 input_shape=(3,)로 사용할 수 있다. 
>>아직 이해되지 않지만, 이후에 추가 설명
input_dim을 사용하면, 다차원 데이터를 수용하지 못한다.
input_shape를 하게 되면, (1,10,10,3)일 때 none자리인 1을 뻬고, 10,10,3을 넣어주면 된다. 

'''