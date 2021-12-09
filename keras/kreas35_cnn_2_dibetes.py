from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Conv2D, Flatten ,MaxPooling2D
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
import numpy as np

#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target
print(datasets.feature_names)

import pandas as pd
xx = pd.DataFrame(x, columns=datasets.feature_names)
# print(type(xx))
# print(xx.corr())
# xx['sick'] = y
# import matplotlib.pyplot as plt
# import seaborn as sns
# plt.figure(figsize=(10,10))
# sns.heatmap(data=xx.corr(), square=True, annot=True, cbar=True)
# plt.show()

x = xx.drop(['sex'], axis =1)
x = x.to_numpy()

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, random_state=42)
print(x_train.shape) # (309, 9)
print(x_test.shape) # (133, 9)
# print(x.shape) (442, 9)

scaler = MinMaxScaler()         
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(309, 3, 3, 1)
x_test = x_test.reshape(133, 3, 3, 1)

#2. 모델구성
model = Sequential() 
model.add(Conv2D(7, kernel_size = (2,2),input_shape = (3,3,1)))                      
model.add(Dropout(0.2))       
model.add(Flatten()) 
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(16, activation='relu'))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', patience=50, mode='min', verbose=1) 
mcp = ModelCheckpoint(monitor= 'val_loss', mode = 'min', verbose =1, save_best_only=True,
                      filepath = './_ModelCheckPoint/keras27_2_MCP.hdf5')
model.fit(x_train, y_train, epochs=200, batch_size=32,
          validation_split=0.3, callbacks=[es, mcp])
model.save('./_save/keras27_2_save_model.h5') 

#4. 평가, 예측
print ('====================== 1. 기본출력 ========================')
loss = model.evaluate(x_test, y_test)
print('loss:', loss)

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2 스코어:', r2)



print ('====================== 2. load_model 출력 ========================')
model2 = load_model('./_save/keras27_2_save_model.h5')

loss2 = model2.evaluate(x_test, y_test)
print('loss:', loss2)

y_predict2 = model2.predict(x_test)
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict2)
print('r2 스코어:', r2)

print ('====================== 3. ModelCheckPoint 출력 ========================')
model3 = load_model('./_ModelCheckPoint/keras27_2_MCP.hdf5')

loss3 = model3.evaluate(x_test, y_test)
print('loss:', loss3)

y_predict3 = model3.predict(x_test)
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2 스코어:', r2)

