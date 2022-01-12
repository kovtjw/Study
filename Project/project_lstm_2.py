import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense,LSTM, Input, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import mean_squared_error as MSE
import matplotlib.pyplot as plt
path = '../_data/project data/' 
gwangju = pd.read_csv(path +"data_광주.csv")

dataset = gwangju.drop(['일자'],axis = 1)
bb = dataset['가격']
dataset = np.array(dataset)

x1 = gwangju.drop(['일자','가격'],axis = 1)
x1 = x1[-21:]  # (21, 4)
x1 = np.array(x1)
x1 = x1.reshape(1,21,4)
print(x1.shape)
# print(dataset.shape)  # (1458, 5)
# print(dataset.info())


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

print(x,y)
print(x.shape)  # (1435, 21, 4)
# print(y)  # (1431, 7)
print(y.shape)  # (1435, 3)

x_train, x_test, y_train, y_test = train_test_split(x,y,
        train_size =0.7, shuffle=True, random_state = 100)



# mms = MinMaxScaler()
# X_train = mms.fit_transform(x_train)
# X_test = mms.transform(x_test)
# print(x_train.shape, y_train.shape)  # (1001, 21, 4) (1001, 7)


#2. 모델구성
model = Sequential()
model.add(LSTM(32, activation='relu',input_shape = (21, 4)))
# model.add(Dense(130))
model.add(Dense(32,activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(8,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7))

# model.summary()

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer = 'adam')
model.fit(x_train, y_train, epochs=20, batch_size=2,
          validation_split=0.2)

#4. 평가, 예측
loss = model.evaluate(x_test,y_test)
print("loss : ",loss)

y_pred = model.predict(x1)
print('일주일치의 감자 예측 가격:', y_pred)  # (1454, 2, 4)

y_predict = model.predict(x)

aa = []
for i in y_predict:
    aa.append(i[-1])

aa = np.array(aa)
bb = bb[20:-7]
print(aa.shape)
print(bb.shape)

plt.figure(figsize=(12, 9))
plt.plot(bb, label = 'actual')
plt.plot(aa, label = 'prediction')
plt.legend()
plt.show()