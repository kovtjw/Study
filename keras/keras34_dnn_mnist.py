from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
from tensorflow.keras.utils import to_categorical

#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(60000,28,28,1) 
x_test = x_test.reshape(10000, 28,28,1)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(x_train.shape,x_test.shape)
print(y_train.shape,y_test.shape)


scaler = StandardScaler()

n = x_train.shape[0]# 이미지갯수 50000
x_train_reshape = x_train.reshape(n,-1) 
x_train = scaler.fit_transform(x_train_reshape) 
# x_train = x_train_transe.reshape(x_train.shape) 

m = x_test.shape[0]
x_test = scaler.transform(x_test.reshape(m,-1)) #.reshape(x_test.shape)

#2. 모델구성
model = Sequential()
model.add(Dense(100, input_shape=(784, )))
# model.add(Dense(64, input_shape=(784, )))  # 위와 동일함
model.add(Dense(80, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(130, activation='sigmoid'))
model.add(Dense(80, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(5))
model.add(Dense(10, activation = 'softmax'))


#3. 컴파일, 훈련
import time
model.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])
start = time.time()
model.fit(x_train, y_train, epochs=16, batch_size=16,
          validation_split=0.3)
end = time.time()



#4. 평가, 예측
print ('====================== 1. 기본출력 ========================')
loss = model.evaluate(x_test, y_test)
print('loss:', loss)

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2 스코어:', r2)

'''
====================== 1. 기본출력 ========================
313/313 [==============================] - 0s 940us/step - loss: 0.1612 - accuracy: 0.9574
loss: [0.16115860641002655, 0.9574000239372253]
'''