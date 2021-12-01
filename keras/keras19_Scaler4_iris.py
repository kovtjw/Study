from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
import numpy as np

#1. 데이터

datasets = load_iris()
x = datasets.data
y = datasets.target
from tensorflow.keras.utils import to_categorical
y = to_categorical(y)

x_train, x_test, y_train, y_test = train_test_split(x,y,
        train_size =0.8, shuffle=True, random_state = 46)


scaler = RobustScaler()         
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


#2. 모델구성
model = Sequential()
model.add(Dense(50,activation='relu', input_dim=4))
model.add(Dense(70, activation= 'sigmoid')) 
model.add(Dense(50, activation= 'linear'))
model.add(Dense(50, activation= 'linear'))
model.add(Dense(50, activation= 'sigmoid'))
model.add(Dense(3, activation= 'softmax'))

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])
 
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=10, mode='auto',
                   verbose=1, restore_best_weights=False) # restore_best_weights=True : 최종값 이전에 가장 좋았던 값 도출함

model.fit(x_train, y_train, epochs=100, batch_size=5,
          validation_split=0.3, callbacks=[es])

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss :', loss[0])
print('accuracy :', loss[1])





'''
# 결과 
그냥 : loss : 0.07094156742095947

MinMax :  loss : 0.19691327214241028
 
Standard : loss : 0.14737610518932343

Robuster : loss : 0.14516735076904297

MaxAbs : loss : 0.1305277794599533
'''

'''
# relu 결과 

그냥 : loss : 0.1386108696460724

MinMax : loss : 0.1606500893831253
 
Standard : loss : 0.10095973312854767

Robuster : loss : 0.27302366495132446

MaxAbs : loss : 0.12254985421895981

'''