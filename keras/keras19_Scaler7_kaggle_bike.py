import numpy as np
import pandas as pd 
from sklearn.metrics import r2_score, mean_squared_error #mse
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler

def RMSE(y_test, y_pred):
    return np.sqrt(mean_squared_error(y_test, y_pred))

#1. 데이터 
path = './_data/bike/'   # '..'의 뜻은 이전 단계이다. / '.'은 현재 단계 >> 여기선 STUDY 폴더
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
        train_size =0.9, shuffle=True, random_state = 26)


scaler = MinMaxScaler()         
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
test_file = scaler.transform(test_file)



#2. 모델구성
model = Sequential()
model.add(Dense(233, input_dim=8)) 
model.add(Dense(144,activation='relu'))
model.add(Dense(89))
model.add(Dense(55))
model.add(Dense(34))
model.add(Dense(144))
model.add(Dense(89))
model.add(Dense(8))
model.add(Dense(5))
model.add(Dense(3))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer = 'adam')
 
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=100, mode='auto',
                   verbose=1, restore_best_weights=False)

model.fit(x_train, y_train, epochs=1000, batch_size=0,
          validation_split=0.3, callbacks=[es])

#4. 평가, 예측
loss = model.evaluate(x_test,y_test)
print("loss : ",loss)

y_pred = model.predict(x_test)

r2 = r2_score(y_test, y_pred)
print ('r2 :', r2)

rmse = RMSE(y_test,y_pred)
print('RMSE : ', rmse)



##################### 제출용 제작 ####################
results = model.predict(test_file)

submit_file ['count'] = results

# print(submit_file[:10])

submit_file.to_csv(path + 'StandardScaler.csv', index=False) # to_csv하면 자동으로 인덱스가 생기게 된다. > 없어져야 함

'''
# 결과 
그냥  
loss :  1.5799766778945923
r2 : 0.22983952257855256
RMSE :  1.2569712357015561

MinMax  
loss :  1.561394453048706
r2 : 0.24867304284023162
RMSE :  1.2495576774748958
 
Standard  
loss :  1.557446837425232
r2 : 0.25057256560251673
RMSE :  1.247977097041303

Robuster  
loss :  1.555137276649475
r2 : 0.25168395201165583
RMSE :  1.2470513907484473


MaxAbs 
loss :  1.5557773113250732
r2 : 0.25137591312360397
RMSE :  1.2473080341819072
'''

'''
# relu 결과 

그냥 

loss :  1.4541757106781006
r2 : 0.30026556509421853
RMSE :  1.2058920857196693

MinMax 
loss :  1.429654598236084
r2 : 0.3120648837439437
RMSE :  1.1956816412179956
 
Standard 
loss :  1.4192843437194824
r2 : 0.31705487879476224
RMSE :  1.1913372602742838


Robuster
loss :  1.4265260696411133
r2 : 0.3135701906283461
RMSE :  1.1943727579828665


MaxAbs 
loss :  1.4350309371948242
r2 : 0.30947787544154337
RMSE :  1.1979277365636631
'''
