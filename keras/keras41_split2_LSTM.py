import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM

a = np.array(range(1,101))  
size = 5  # 5개씩 자른다.
x_predict = np.array(range(96,106))   
# x_predict = x_predict.reshape(10,1,1)
# print(x_predict.shape)

def split_x(data, size):
    aaa = []
    for i in range(len(data) - size +1): # 1~10 - 5 +1 //size가 나오는 가장 마지막 것을 생각해서 +1을 해주어야 함
        subset = data[i : (i+size)]      # [1 : 6] = 1,2,3,4,5 
        aaa.append(subset)                  # 1,2,3,4,5에 []를 붙인다.
    return np.array(aaa)                    # numpy 배열로 리턴한다.

dataset = split_x(a, size)                  # 대입했을 때
x_predict = split_x(x_predict,5)
# print(bbb.shape)

x = dataset[:,:-1]  # :, = 0:-1
y = dataset[:,-1]   
x_predict = x_predict[:,:-1]
# x_pred2 = ccc[:,-1]

# print(x,y)
# print(x_predict.shape) # 6,4
x = x.reshape(96,4,1)
x_predict = x_predict.reshape(6,4,1)

print(x_predict)
# print(x.shape,y.shape) #(96, 4, 1) (96,)
# print(x_pred1.shape,x_pred2.shape) # (6, 4) (6,)
# print(x.shape) # (96, 4, 1)
# (6, 4) (6,)

#2. 모델구성
model = Sequential()
model.add(LSTM(32,activation='relu',input_shape = (4,1))) # input_shape 할 때 행은 넣지 않는다. 
model.add(Dense(128, activation='relu'))       # 모든 activation은 다음으로 보내는 값을 한정 시키기 때문에 윗 줄에서도 가능하다.
model.add(Dense(64, activation='relu'))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(1))

#3. 컴파일, 훈련 
model.compile(loss = 'mae', optimizer = 'adam')   # optimizer는 loss값을 최적화 한다.
model.fit(x, y, epochs = 200)

#4. 평가, 예측 
model.evaluate(x, y)
result = model.predict(x_predict)
print(result)

'''
[[[ 96]
  [ 97]
  [ 98]
  [ 99]]

 [[ 97]
  [ 98]
  [ 99]
  [100]]

 [[ 98]
  [ 99]
  [100]
  [101]]

 [[ 99]
  [100]
  [101]
  [102]]

 [[100]
  [101]
  [102]
  [103]]

 [[101]
  [102]
  [103]
  [104]]]
'''
'''
[[100.89785]
 [101.8796 ]
 [102.85959]
 [103.83774]
 [104.81401]
 [105.78838]]
'''