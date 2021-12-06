from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np 
from sklearn.model_selection import train_test_split
import time
from sklearn.datasets import load_boston
from tensorflow.python.keras.saving.save import load_model

#1. 데이터
datasets = load_boston()

x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x,y,
        train_size =0.8, shuffle=True, random_state = 46)

# 2. 모델구성
model = Sequential()
model.add(Dense(100, input_dim=13)) 
model.add(Dense(80))
model.add(Dense(130))
model.add(Dense(80))
model.add(Dense(5))
model.add(Dense(1))


# 모델에 대한 정의나 구성은 살아 있어야 한다. model = Sequential() 제외 시 사용 불가

# model.summary()
# model.save_weights('./_save/keras25_1_save_weights.h5') > 훈련되기 전에 랜덤값을 넣어서 연산 시켰다


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
# start = time.time()
# hist = model.fit(x_train, y_train, epochs=100, batch_size=5,
#           validation_split=0.3)
# end = time.time() - start
# print('걸린시간:', round(end,3), '초')
 
# model.save('./_save/keras25_3_save_model.h5')  ##
# model.save_weights('./_save/keras25_1_save_weights.h5')
model.load_weigths('./_save/keras25_3_save_model.h5') 


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss:', loss)

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2 스코어:', r2)


'''
모델을 구성해주고, 컴파일 넣어주고, 로드 웨이트를 사용하면 된다.
저장된 로드 웨이트는 가장 좋은 가중치일까?
명확하게 말한다면, 지정한 에포만큼 돌렸을 때의 가중치이기 때문에 가장 좋다고 말할 수는 없지만

얼리스탑에서 가장 좋은 가중치를 찾는 방법은??

'''