import numpy as np 
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split

#1. 데이터
datasets = fetch_covtype()
x = datasets.data 
y = datasets.target
print(x.shape,y.shape) # (581012, 54) (581012,)
print(np.unique(y)) # [1 2 3 4 5 6 7]

from tensorflow.keras.utils import to_categorical
y = to_categorical(y)

x_train, x_test, y_train, y_test = train_test_split(x,y,
        train_size =0.8, shuffle=True, random_state = 36)

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

#2. 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(233, input_dim=54)) 
model.add(Dense(144))
model.add(Dense(55))
model.add(Dense(21))
model.add(Dense(8, activation= 'softmax'))

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor = 'val_loss', patience=10, mode='auto',
                   verbose =1, restore_best_weights=False)

model.fit(x_train, y_train, epochs =100, batch_size=1000,
          validation_split=0.3, callbacks=[es])

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss :', loss[0])
print('accuracy : ', loss[1])



#batch_size 디폴트 값은 32이다.
# 출처 : https://stats.stackexchange.com/questions/153531/what-is-batch-size-in-neural-network