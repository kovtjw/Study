from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes # diabete 당뇨병


#1. 데이터 
datasets = load_diabetes()
x = datasets.data
y = datasets.target 

x_train, x_test, y_train, y_test = train_test_split(x,y,
                train_size=0.6, shuffle=True, random_state=49)



#2. 모델구성
model = Sequential()
model.add(Dense(13, input_dim=10)) 
model.add(Dense(12))
model.add(Dense(11))
model.add(Dense(10))
model.add(Dense(9))
model.add(Dense(8))
model.add(Dense(7))
model.add(Dense(6))
model.add(Dense(5))
model.add(Dense(4))
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

model.fit(x_train,y_train, epochs=200, batch_size=1)
# print(x.shape, y.shape) # (442,10) , (442, )

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss :', loss)

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2 스코어 :', r2)


# 과제 
# R2 : 0.62 이상

'''









'''