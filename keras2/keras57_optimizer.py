import numpy as np

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,3,5,7,6,7,11,8,9,6])

#2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(1000, input_dim = 1))
model.add(Dense(1000))
model.add(Dense(1000))
model.add(Dense(1))

#3. 컴파일, 훈련
from tensorflow.keras.optimizers import Adam, Adadelta, Adagrad, Adamax
from tensorflow.keras.optimizers import RMSprop, SGD, Nadam

learning_rate = 0.01

# optimizer = Adam(learning_rate=learning_rate)
# loss : 4.4591 lr : 1e-05 결과물 : [[11.057547]]
optimizer = Adadelta(learning_rate=learning_rate)
# loss : 4.4213 lr : 0.01 결과물 : [[11.37494]]
# optimizer = Adagrad(learning_rate=learning_rate)
# loss : 4.5137 lr : 0.0001 결과물 : [[11.440412]]
# optimizer = Adamax(learning_rate=learning_rate)
# loss : 3.9232 lr : 0.0001 결과물 : [[11.077569]]
# optimizer = RMSprop(learning_rate=learning_rate)
# loss : 4.4209 lr : 1e-05 결과물 : [[11.300552]]
# optimizer = SGD(learning_rate=learning_rate)
# loss : 4.5349 lr : 1e-05 결과물 : [[11.424333]]
# optimizer = Nadam(learning_rate=learning_rate)
# loss : 4.4467 lr : 1e-05 결과물 : [[11.114904]]

# model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mse'])
model.compile(loss = 'mse', optimizer = optimizer)
model.fit(x,y, epochs = 100, batch_size = 1)

#4. 평가, 예측
loss = model.evaluate(x,y, batch_size=1)
y_pred = model.predict([11])
print('loss :', round(loss,4),  'lr :', learning_rate,'결과물 :', y_pred)