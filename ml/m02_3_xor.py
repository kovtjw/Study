import numpy as np
from sklearn.svm import LinearSVC
from sklearn.linear_model import Perceptron   ## 원조
from sklearn.metrics import accuracy_score


###### 인공지능의 겨울  XOR #######


#1. 데이터   ### xor 데이터 ### 
x_data = [[0,0],[0,1],[1,0],[1,1]]
y_data = [0, 1, 1, 0]

#2. 모델
# model = LinearSVC()
model = Perceptron()     # xor에 막혔다. 

#3. 훈련
model.fit(x_data, y_data)

#4. 평가, 예측
y_pred = model.predict(x_data)
print(x_data, " 의 예측결과 : ", y_pred)
results = model.score(x_data, y_data)
print('model.score : ', results)

acc = accuracy_score(y_data, y_pred)
print('accuracy_score : ', acc)