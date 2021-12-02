# import tensorflow as tf 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense 
import numpy as np 

#1. 데이터 (정제된 데이터)
x = np.array([1,2,3])
y = np.array([1,2,3])


#2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim=1)) 
model.add(Dense(3, activation='relu'))
model.add(Dense(4, activation='sigmoid'))
model.add(Dense(2, activation='relu'))
model.add(Dense(1))
model.summary()

'''
#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam') 

model.fit(x, y, epochs=30, batch_size=3) 

#4. 평가, 예측 
loss = model.evaluate(x,y) 
print('loss :', loss)
result = model.predict([4])
print('4의 예측값 :', result)

'''

'''
파라미터를 계산할 때에는 
model = Sequential()
model.add(Dense(5, input_dim=1)) 
model.add(Dense(3))
model.add(Dense(4))
model.add(Dense(2))
model.add(Dense(1))

> 기본적으로 생각 할 때 42번의 연산이라고 생각하지만, 실제 연산된 갯수는 57번이었다. 
>> 15번의 차이가 나는 이유는 'b, 바이어스'의 연산이 추가되기 때문이다. 

# batch_size도 성능에 영향을 미친다. 
'''
 
'''
dfsdfsa
'''
