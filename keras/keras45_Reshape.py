from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, Reshape, Conv1D, LSTM
from tensorflow.keras.datasets import mnist

#1. 데이터





#2. 모델 구성
model = Sequential() 
model.add(Conv2D(10, kernel_size = (2,2), strides =1,
                 padding = 'same',input_shape = (28,28,1))) 
model.add(MaxPooling2D())  
model.add(Conv2D(5,(2,2), activation = 'relu'))   # 13, 13, 5
model.add(Conv2D(7, (2,2), activation='relu'))    # 12, 12, 7
model.add(Conv2D(7, (2,2), activation='relu'))    # 12, 12, 7
model.add(Conv2D(10, (2,2), activation='relu'))    # 12, 12, 7
model.add(Flatten())                              # (None, 1000)
model.add(Reshape(target_shape= (100,10)))                      # (None, 100, 10)
model.add(Conv1D(5,2))
model.add(LSTM(15))
model.add(Dense(10,activation='softmax'))
model.summary()

#input_shape = (a,b,c) > kernel_size = (d,e) = (a-d+1, b-e+1)  // 

# model.add(Conv2D(5, (3,3), activation='relu'))                     # 3,3,5
# model.add(Dropout(0.2))
# model.add(Conv2D(7, (2,2), activation='relu'))                     # 2,2,7
# model.add(Flatten())  # flatten (Flatten)            (None, 252)               0
# model.add(Dense(64))
# model.add(Dropout(0.2))
# model.add(Dense(16))
# model.add(Dense(5, activation = 'softmax'))




