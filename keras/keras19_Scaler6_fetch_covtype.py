import numpy as np 
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler

#1. 데이터
datasets = fetch_covtype()
x = datasets.data 
y = datasets.target

import pandas as pd 
y = pd.get_dummies(y)
print(x.shape,y.shape)

x_train, x_test, y_train, y_test = train_test_split(x,y,
        train_size =0.8, shuffle=True, random_state = 36)


scaler = MaxAbsScaler()         
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)



#2. 모델구성
model = Sequential()
model.add(Dense(233, input_dim=54)) 
model.add(Dense(144,activation='relu'))
model.add(Dense(55))
model.add(Dense(21))
model.add(Dense(7, activation= 'softmax'))

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor = 'val_loss', patience=10, mode='auto',
                   verbose =1, restore_best_weights=False)

model.fit(x_train, y_train, epochs =100, batch_size=1000,
          validation_split=0.3, callbacks=[es])

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss :', loss[0])
print('accuracy : ', loss[1])



'''
# 결과 
그냥 : loss : 0.6739720106124878

MinMax : loss : 0.6426326632499695
 
Standard : loss : 0.6362609267234802

Robuster : loss : 0.6357030272483826

MaxAbs :  loss : 0.6391305327415466
'''

'''
# relu 결과 

그냥 : loss : 0.6491599082946777

MinMax : loss : 0.3096392750740051
 
Standard : loss : 0.31676140427589417

Robuster : loss : 0.3119511604309082

MaxAbs : loss : 0.3365333080291748

'''