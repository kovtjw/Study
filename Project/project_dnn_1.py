import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler,MaxAbsScaler, LabelEncoder
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score

path = '../_data/project data/' 
gwangju = pd.read_csv(path +"merge_gwangju.csv")


print(gwangju.info())


x = gwangju.drop(['일자','가격'], axis = 1)
y = gwangju['가격']

print(x.shape)     # (1458, 4)

print(y.shape)     # (1458,)


x_train, x_test, y_train, y_test = train_test_split(x,y,
        train_size= 0.8, shuffle = True, random_state=42)

print(x_train.shape)     # (1458, 4)
print(y_train.shape)     # (1458,)

# scaler = MinMaxScaler()         
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)


# 2. 모델구성
model = Sequential()
model.add(Dense(64, input_dim=5)) 
model.add(Dropout(0.5)) 
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5)) 
model.add(Dense(16, activation='relu'))
model.add(Dense(1))

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

#3. 컴파일, 훈련
model.compile(loss='mae', optimizer = 'adam')
es = EarlyStopping(monitor='val_loss', patience=5, mode='auto',
                   verbose=1, restore_best_weights=True)
model.fit(x_train, y_train, epochs=100, batch_size=2,
          validation_split=0.1, callbacks=[es])

#4. 평가, 예측
print ('====================== 1. 기본출력 ========================')
loss = model.evaluate(x_test, y_test)
y_pred = model.predict(x_test)
# y_pred = y_pred.round(0).astype(int)
print(y_pred[-1])

# plt.scatter(x,y) # 데이터를 점으로 흩뿌린다.
# plt.plot(y_pred, color='red') # 연속된 선을 그려준다.
# plt.show()

 # 데이터를 점으로 흩뿌린다.
plt.plot(y_pred, color='red') # 연속된 선을 그려준다.
plt.title('감자 가격')
plt.xlabel('금액')
plt.ylabel('random.randn')

plt.show()