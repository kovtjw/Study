###########################################
# 각각의 Scaler의 특성과 정의 정리해 놓을 것
###########################################
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
import numpy as np

#1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets.target
# print(np.min(x), np.max(x)) # 0.0 711.0


# print(x.shape)      #(506, 13): 컬럼이 13개
# x = x/711.              # .을 쓰는 이유는 부동소수점으로 나눈다는 뜻
# x = x/np.max(x)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, random_state=42
)


scaler =  MaxAbsScaler()         
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)



#2. 모델구성
model = Sequential()
model.add(Dense(100, input_dim=13)) 
model.add(Dense(80, activation='relu'))
model.add(Dense(130, activation='sigmoid'))
model.add(Dense(80, activation='relu'))
model.add(Dense(5))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=100, mode='min', verbose=1) 

model.fit(x_train, y_train, epochs=10000, batch_size=32,
          validation_split=0.3, callbacks=[es])

model.save('./_save/keras19_1_save_model.h5') 

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss:', loss)

y_predict = model.predict(x_test)


'''
# scaler 결과 
그냥 : loss: 19.14067268371582

MinMax : loss: 16.720535278320312
 
Standard : loss: 17.38316535949707

Robuster : loss: 15.025249481201172

MaxAbs : loss: 20.80170440673828
'''

'''
# relu 결과 

그냥 : loss: 15.611274719238281

MinMax : loss: 8.855862617492676
 
Standard : loss: 12.234639167785645

Robuster :loss: 9.609686851501465

MaxAbs : loss: 10.25717544555664

'''

'''
scaler 종류별 정의 및 비교

1. Scaler - MinMaxScaling

1) 데이터를 0과 1사이의 값으로 반환
2) (x-x의 최소값)/(x의 최대값 - x의 최소값)
3) 데이터의 최소, 최대 값을 알 경우 사용

2. Scaler - StandardScaler

1) 기존 변수에 범위를 정규 분포로 변환
2) (x-x의 평균값) / (x의 표준편차)
3) 데이터의 최소, 최대 값을 모를 경우 사용

3. Scaler - RobustScaler

1) StandardScaler에 의한 표준화보다 동일한 값을 더 넓게 분포
2) 이상치(outliner)를 포함하는 데이터를 표준화 하는 경우 사용

4. Scaler - MaxAbscaler

최대 절대값과 0이 각 1,0이 되도록 하여 양수 데이터로만 구성되게 스케일링하는 기법이다.
minmaxscaler과 유사하지만 음수와 양수값에 따른 대칭 분포를 유지하게 되는 특징이 있다.

자세한 내용은 블로그에 정리함 
https://blog.naver.com/kovtjw/222584175176
'''
