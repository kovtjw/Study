from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
import numpy as np

#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target
# print(np.min(x), np.max(x))

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=42
)


scaler = MaxAbsScaler()         
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)



#2. 모델구성
model = Sequential()
model.add(Dense(100, input_dim=10)) 
model.add(Dense(80,activation='relu'))
model.add(Dense(130))
model.add(Dense(80))
model.add(Dense(5))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=50, mode='min', verbose=1) 
model.fit(x_train, y_train, epochs=200, batch_size=32,
          validation_split=0.3, callbacks=[es])
model.save('./_save/keras25_3_save_model.h5') 

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss:', loss)




'''
# 결과 
그냥 : loss: 3310.6396484375

MinMax : loss: 3349.428466796875
 
Standard : loss: 3379.4072265625

Robuster : loss: 3601.710693359375

MaxAbs : loss: 3777.190673828125
'''

'''
# relu 결과 

그냥 : loss: 4003.416015625

MinMax : loss: 4382.30419921875
 
Standard : loss: 5160.57666015625

Robuster : loss:4017.225830078125

MaxAbs : loss: 4201.33203125

'''