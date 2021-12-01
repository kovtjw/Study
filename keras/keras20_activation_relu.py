# import tensorflow as tf 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense 
import numpy as np 

#1. 데이터 (정제된 데이터)
x = np.array([1,2,3])
y = np.array([1,2,3])


#2. 모델구성
model = Sequential()
model.add(Dense(300, input_dim=1)) 
model.add(Dense(150, activation='relu'))
model.add(Dense(160, activation='sigmoid'))
model.add(Dense(105, activation='relu'))
model.add(Dense(75))
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam') 

model.fit(x, y, epochs=30, batch_size=3) 

#4. 평가, 예측 
loss = model.evaluate(x,y) 
print('loss :', loss)
result = model.predict([4])
print('4의 예측값 :', result)


'''
'activation'은 레이어의 결과 값을 다음 레이어로 전달할 때 '한정'시키는 역할을 한다. 
모델 구성의 연산 중 -값을 뺐더니, 결과 값이 조금 더 좋게 도출되었다. 
relu : 결과 값을 다음 레이어로 전달 할 때 -값을 제외하고 양수만으로 한정시키는 역할을 함 
y=relu(wx+b) >> '하이퍼 파라미터 튜닝'의 일부 

'''

 
