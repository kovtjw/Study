import numpy as np
import pandas as pd 
import datetime
import time
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense,LSTM, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

path = '../_data/project data/' 
gwangju = pd.read_csv(path +"광주 .csv")

dataset = gwangju.drop(['일자'],axis = 1)
dataset = np.array(dataset)

def split_xy(dataset, time_steps, y_column):
    x, y = list(), list()
    for i in range(len(dataset)):
        x_end_number = i + time_steps
        y_end_number = x_end_number + y_column
        
        if y_end_number > len(dataset):
            break
        tmp_x = dataset[i:x_end_number,1: ]
        tmp_y = dataset[x_end_number : y_end_number, 0]
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)

x, y = split_xy(dataset, 21, 7)
print(y.shape)  # (1431, 7)
x_train, x_test, y_train, y_test = train_test_split(x,y,
        train_size =0.7, shuffle=True, random_state = 42)


print(x_train.shape, y_train.shape)  # (1001, 21, 4) (1001, 7)
#2. 모델구성
model = Sequential()
model.add(LSTM(32,activation='relu',input_shape = (21,4)))
# model.add(Dense(130))
model.add(Dense(100))
model.add(Dense(130))
model.add(Dense(80))
model.add(Dense(5))
model.add(Dense(7,activation='softmax'))

model.summary()

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer = 'adam')
model.fit(x_train, y_train, epochs=200, batch_size=2,
          validation_split=0.3)

#4. 평가, 예측
loss = model.evaluate(x_test,y_test)
print("loss : ",loss)

y_pred = model.predict(x_test)
print(y_pred)
