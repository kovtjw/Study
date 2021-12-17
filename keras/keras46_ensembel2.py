#1. 데이터 
import numpy as np
import time
x1 = np.array([range(100), range(301,401)])  # 삼성전자 저가, 고가
x2 = np.array([range(101,201),range(411,511),range(100,200)]) # 미국선물 시가, 고가, 종가
x1 = np.transpose(x1) # 의도는 (100,3) 이기 때문에
x2 = np.transpose(x2)

y1 = np.array(range(1001,1101))   # 삼성전자 종가
y2 = np.array(range(101,201))

print(x1.shape,x2.shape, y1.shape,y2.shape)  # (100, 2) (100, 3) (100,)


from sklearn.model_selection import train_test_split
x1_train, x1_test,x2_train, x2_test, y1_train, y1_test,y2_train, y2_test = train_test_split(
    x1,x2, y1,y2, train_size=0.7, random_state=42)

print(x1_train.shape,x2_train.shape, y1_train.shape) # (70, 2) (70, 3) (70,)
print(x1_test.shape,x2_test.shape, y1_test.shape)  # (30, 2) (30, 3) (30,)
print(x1_train.shape,x2_train.shape, y2_train.shape) # (70, 2) (70, 3) (70,)
print(x1_test.shape,x2_test.shape, y2_test.shape) # (30, 2) (30, 3) (30,)


#2. 모델구성
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input


#2-1 모델1
input1 = Input(shape=(2,))   # (70,2) 
dense1 = Dense(5, activation='relu', name = 'dense1')(input1)
dense2 = Dense(7, activation='relu', name = 'dense2')(dense1)
dense3 = Dense(7, activation='relu', name = 'dense3')(dense2)
output1 = Dense(7, activation='relu', name = 'output1')(dense3)


#2-2 모델2
input2 = Input(shape=(3,))   # (70,3) 
dense11 = Dense(10, activation='relu', name = 'dense11')(input2)
dense12 = Dense(10, activation='relu', name = 'dense12')(dense11)
dense13 = Dense(10, activation='relu', name = 'dense13')(dense12)
dense14 = Dense(10, activation='relu', name = 'dense14')(dense13)
output2 = Dense(5, activation='relu', name = 'output2')(dense14)

from tensorflow.keras.layers import Concatenate, concatenate  
merge1 = Concatenate()([output1,output2]) 


# 2-3 output모델1
output21 = Dense(7)(merge1)
output22 = Dense(11)(output21)
output23 = Dense(11, activation='relu')(output22)
last_output1 = Dense(1)(output23)  # y의 열의 갯수

# 2-4 output모델2
output31 = Dense(7)(merge1)
output32 = Dense(11)(output31)
output33 = Dense(11, activation='relu')(output32)
output34 = Dense(11, activation='relu')(output33)
last_output2 = Dense(1)(output34)




# merge2 = Dense(10, activation='relu')(merge1)
# merge3 = Dense(7)(merge2)
# last_output = Dense(1)(merge3)

model = Model(inputs = [input1,input2], outputs= ([last_output1,last_output2]))

model.summary()

#3. 컴파일, 훈련 
start = time.time()

model.compile(loss = 'mae', optimizer = 'adam',metrics=['mse'])   
model.fit([x1_train,x2_train],[y1_train,y2_train],epochs = 20)
end = time.time()- start

#4. 평가, 예측 
result = model.evaluate([x1_test,x2_test],[y1_test,y2_test])
print('loss:', result[0]) 
print('mse:',result[1])
print('1:',result[2])
print('2:',result[3])
print('3:',result[4])


# loss: [809.4212036132812, 793.8211059570312, 15.600076675415039, 630323.9375, 336.79296875]

'''
ensemble
두 개의 x 데이터의 열의 갯수는 달라도 되지만, 행의 갯수는 동일하게 해야 한다. 
레이어의 갯수는 달라도 상관 없다.

'''
'''
loss: 935.4859619140625 : y1,y2의 loss의 총합
mse: 906.9968872070312 : y1의 loss
1: 28.489055633544922 : y2의 loss
2: 822898.375  : y1의 mse
3: 976.9550170898438 : y2의 mse

'''