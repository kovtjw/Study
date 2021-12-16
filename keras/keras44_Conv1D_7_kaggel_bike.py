import numpy as np
import pandas as pd 
from sklearn.metrics import r2_score, mean_squared_error #mse
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv1D
import time

def RMSE(y_test, y_pred):
    return np.sqrt(mean_squared_error(y_test, y_pred))


#1. 데이터 
path = '../_data/kaggle/bike/'  
train = pd.read_csv(path+'train.csv')  
# print(train)      # (10886, 12)
test_file = pd.read_csv(path+'test.csv')
# print(test.shape)    # (6493, 9)
submit_file = pd.read_csv(path+ 'sampleSubmission.csv')
x = train.drop(['datetime', 'casual','registered','count'], axis=1) # axis=1 컬럼 삭제할 때 필요함
test_file = test_file.drop(['datetime'], axis=1) 

y = train['count']

x_train, x_test, y_train, y_test = train_test_split(x,y,
        train_size =0.9, shuffle=True, random_state = 42)

x_train = x_train.values.reshape(9797,4,2)
x_test = x_test.values.reshape(1089,4,2)

#2. 모델구성
model = Sequential()
model.add(Conv1D(32,2,activation='relu',input_shape = (4,2))) 
model.add(Dense(80, activation='relu'))
model.add(Dense(130, activation='sigmoid'))
model.add(Dense(80, activation='relu'))
model.add(Dense(5))
model.add(Dense(1, activation = 'softmax'))

#3. 컴파일, 훈련
start = time.time()

model.compile(loss='mae', optimizer = 'adam')
model.fit(x_train, y_train, epochs=20, batch_size=32,
          validation_split=0.3)
end = time.time()- start


#4. 평가, 예측
loss = model.evaluate(x_test,y_test)
print("loss : ",loss)

y_pred = model.predict(x_test)

r2 = r2_score(y_test, y_pred)
print ('r2 :', r2)

rmse = RMSE(y_test,y_pred)
print('RMSE : ', rmse)
print("걸린시간 : ", round(end, 3), '초')
'''
loss :  188.8512420654297
걸린시간 :  5.915 초
'''