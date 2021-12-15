import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM

#1. 데이터
x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],
              [5,6,7],[6,7,8],[7,8,9],[8,9,10],
              [9,10,11],[10,11,12],
              [20,30,40],[30,40,50],[40,50,60]])
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])
x_predict = np.array([50,60,70])

print(x.shape,y.shape) # (13, 3) (13,)
x = x.reshape(13,3,1)
print(x.shape) # (13,3,1)
print(x_predict.shape) # (3,)

#2. 모델구성
model = Sequential()
model.add(LSTM(32,activation='relu',input_shape = (3,1))) # input_shape 할 때 행은 넣지 않는다. 
model.add(Dense(128, activation='relu'))       # 모든 activation은 다음으로 보내는 값을 한정 시키기 때문에 윗 줄에서도 가능하다.
model.add(Dense(64, activation='relu'))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(1))

#3. 컴파일, 훈련 
model.compile(loss = 'mae', optimizer = 'rmsprop')   # optimizer는 loss값을 최적화 한다.
model.fit(x, y, epochs = 2000)

#4. 평가, 예측 
model.evaluate(x, y)
result = model.predict([[[50],[60],[70]]])
print(result)


# [[79.98173]]


# return_sequences=True 

''''
return_sequences=True 
otimizer = 'rmsprop'
[[[80.63615]
  [79.50726]
  [79.86177]]]
'''

'''
2021.12.14 과제 
LSTM(Long Short-Term Memory)
- 장단기 메모리 알고리즘
- 나중을 위해 정보를 저장함으로써 오래된 시그널이 점차 소실되는 것을 막아줌

1) 구조 
은닉층의 메모리 셀에 망각 게이트, 입력 게이트, 출력 게이트를 추가하여 불필요한
기억은 지우고, 기억해야 할 것들을 정한다.
1-1) forget gate : '과거 정보를 잊기위한' 게이트, 시그모이드 함수를 적용하면
출력 범위는 0~1이기 때문에 출력이 0이면 이전 상태의 정보는 잊고, 1이면 이전 상태의 정보를 기억한다.
1-2) input gate : '현재 정보를 기억하기 위한' 게이트
1-3) output gate : 최종 결과를 위한 게이트
1-4) cell state : 컨베이어 벨트와 같고, 작은 linear interaction만을 적용시키면서
전체 체인을 계속 구동 시킨다. 정보가 전혀 바뀌지 않고 흐르게만 하는 것은
매우 쉽게할 수 있다. 


2) Hyperbolic Tangent(tanh) 함수
sigmoid의 대체제로 사용될 수 있는 활성화 함수 (sigmoid와 매우 유사함)
sigmoid의 출력 범위는 0에서 1사이인 반면, tanh의 출력범위는 -1에서 1사이이다.
tanh는 출력 범위가 더 넓고 경사면이 큰 범위가 더 크기 때문에 더 빠르게 수렴하여
학습하는 특성이 있다. 
sigmoid의 단점인 vanishing gradient problem 의 문제를 그대로 갖고 있다. 

스케일 역할을 한다. 
현재 정보를 얼마나 더할지

3) LSTM 파라미터 연산법
기존 RNN의 연산법에서 4를 곱한다 >> 4번의 연산을 진행하기 때문에
'''