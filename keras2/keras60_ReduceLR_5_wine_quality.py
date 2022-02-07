import pandas as pd 
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import time
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.utils import to_categorical

#1. 데이터
path = 'D:\\_data\\'
datasets = pd.read_csv(path + 'winequality-white.csv',sep=';', header = 0)
datasets = datasets.values
x = datasets[:, :11]  # 모든 행, 10번째 컬럼까지
y = datasets[:, 11]

y = to_categorical(y)

x_train, x_test, y_train, y_test = train_test_split(x,y,
                    train_size = 0.75, random_state = 66, shuffle = True,
                stratify=y)

scaler =  MinMaxScaler()         
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델구성
model = Sequential()
model.add(Dense(32,activation='relu',input_dim = 11)) 
model.add(Dense(80, activation='relu'))
model.add(Dense(130, activation='sigmoid'))
model.add(Dense(80, activation='relu'))
model.add(Dense(5))
model.add(Dense(2, activation = 'softmax'))

#3. 컴파일, 훈련
import time
from tensorflow.keras.optimizers import Adam
learning_rate = 0.001
optimizer = Adam(learning_rate=learning_rate)
model.compile(loss='categorical_crossentropy', optimizer= optimizer, metrics='mae')
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
es = EarlyStopping(monitor='val_loss', patience=15, mode='auto',
                   verbose=1, restore_best_weights=False)
ReduceLR = ReduceLROnPlateau(monitor='val_loss', patience=5, mode = 'min',
                             verbose=3, factor=0.5)

start = time.time()
model.fit(x_train, y_train, epochs=300, batch_size=2,
          validation_split=0.25, callbacks=[es, ReduceLR])
end = time.time()

#4. 평가, 예측
print ('====================== 1. 기본출력 ========================')
loss = model.evaluate(x_test, y_test)
print('loss:', loss)

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2 스코어:', r2)