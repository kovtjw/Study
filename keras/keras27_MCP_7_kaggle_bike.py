import numpy as np
import pandas as pd 
from sklearn.metrics import r2_score, mean_squared_error #mse
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler

def RMSE(y_test, y_pred):
    return np.sqrt(mean_squared_error(y_test, y_pred))

#1. 데이터 
path = '../_data/kaggle/bike/'   # '..'의 뜻은 이전 단계이다. / '.'은 현재 단계 >> 여기선 STUDY 폴더
train = pd.read_csv(path+'train.csv')  
# print(train)      # (10886, 12)
test_file = pd.read_csv(path+'test.csv')
# print(test.shape)    # (6493, 9)
submit_file = pd.read_csv(path+ 'sampleSubmission.csv')
# print(submit.shape)     # (6493, 2)
# print(submit_file.columns)

x = train.drop(['datetime', 'casual','registered','count'], axis=1) # axis=1 컬럼 삭제할 때 필요함
test_file = test_file.drop(['datetime'], axis=1) 

y = train['count']
# 로그변환
y = np.log1p(y)
# y = np.log(y)

x_train, x_test, y_train, y_test = train_test_split(x,y,
        train_size =0.9, shuffle=True, random_state = 42)


scaler = MinMaxScaler()         
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
test_file = scaler.transform(test_file)



#2. 모델구성
model = Sequential()
model.add(Dense(100, input_dim=8)) 
model.add(Dense(80, activation='relu'))
model.add(Dense(130, activation='sigmoid'))
model.add(Dense(80, activation='relu'))
model.add(Dense(5))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer = 'adam')
 
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', patience=50, mode='auto',
                   verbose=1, restore_best_weights=False)
mcp = ModelCheckpoint (monitor = 'val_loss', mode = 'min', verbose = 1, save_best_only=True,
                       filepath = './_ModelCheckPoint/keras27_7_MCP.hdf5')
model.fit(x_train, y_train, epochs=1000, batch_size=32,
          validation_split=0.3, callbacks=[es,mcp])

model.save('./_save/keras27_7_save_model.h5')

#4. 평가, 예측
loss = model.evaluate(x_test,y_test)
print("loss : ",loss)

y_pred = model.predict(x_test)

r2 = r2_score(y_test, y_pred)
print ('r2 :', r2)

rmse = RMSE(y_test,y_pred)
print('RMSE : ', rmse)

#4. 평가, 예측
print ('====================== 1. 기본출력 ========================')
loss = model.evaluate(x_test, y_test)
print('loss:', loss)

y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)
print('r2 스코어:', r2)

rmse = RMSE(y_test,y_pred)
print('RMSE : ', rmse)

print ('====================== 2. load_model 출력 ========================')
model2 = load_model('./_save/keras27_7_save_model.h5')

loss2 = model2.evaluate(x_test, y_test)
print('loss:', loss2)

y_predict2 = model2.predict(x_test)
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict2)
print('r2 스코어:', r2)

rmse = RMSE(y_test,y_pred)
print('RMSE : ', rmse)

print ('====================== 3. ModelCheckPoint 출력 ========================')
model3 = load_model('./_ModelCheckPoint/keras27_7_MCP.hdf5')

loss3 = model3.evaluate(x_test, y_test)
print('loss:', loss3)

y_predict3 = model3.predict(x_test)

r2 = r2_score(y_test, y_predict)
print('r2 스코어:', r2)

rmse = RMSE(y_test,y_pred)
print('RMSE : ', rmse)
'''
====================== 1. 기본출력 ========================
35/35 [==============================] - 0s 469us/step - loss: 1.4179
loss: 1.4178920984268188
r2 스코어: 0.325472912632533
RMSE :  1.190752760091876
====================== 2. load_model 출력 ========================
35/35 [==============================] - 0s 440us/step - loss: 1.4179
loss: 1.4178920984268188
r2 스코어: 0.325472912632533
RMSE :  1.190752760091876
====================== 3. ModelCheckPoint 출력 ========================
35/35 [==============================] - 0s 499us/step - loss: 1.4375
loss: 1.4375331401824951
r2 스코어: 0.325472912632533
RMSE :  1.190752760091876
'''

'''
데이터가 작고, 교육용으로 잘 만들어져 있기 때문에 
비슷하다고 느낄 수 있지만, 이후에는 체크포인트를 사용하는 것이 더 낫다.


'''

