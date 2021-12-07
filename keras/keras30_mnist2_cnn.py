import numpy as np
from tensorflow.keras.datasets import mnist # 7만장 데이터 가로 28 세로 28의 손글씨 이미지
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Conv2D, Flatten
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
from tensorflow.keras.utils import to_categorical

(x_train, y_train), (x_test, y_test) = mnist.load_data()
# print(x_train.shape, y_train.shape)   # (60000, 28, 28) (60000,)  >> 흑백
# print(x_test.shape, y_test.shape)  # (10000, 28, 28) (10000,)
x_train = x_train.reshape(60000,28,28,1) # reshape는 전체를 다 곱해서 일정하면 상관 없다. (60000,28,14,2)도 가능
x_test = x_test.reshape(10000, 28,28,1)
# print(x_train.shape)
# print(np.unique(y_train, return_counts=True))  # (60000, 28, 28, 1)(array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949],dtype=int64))


y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(y_train.shape) # (60000, 10)
model = Sequential() 
model.add(Conv2D(7, kernel_size = (3,3), input_shape = (28,28,1))) 
model.add(Conv2D(5, (3,3), activation='relu'))                     
model.add(Dropout(0.2))
model.add(Conv2D(4, (2,2), activation='relu'))         
model.add(Flatten()) 
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(16, activation='relu'))
model.add(Dense(10, activation = 'softmax'))



#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])
 
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', patience=5, mode='auto',
                   verbose=1, restore_best_weights=False) # restore_best_weights=True : 최종값 이전에 가장 좋았던 값 도출함
mcp = ModelCheckpoint (monitor = 'val_loss', mode = 'min', verbose = 1, save_best_only=True,
                       filepath = './_ModelCheckPoint/keras27_5_MCP.hdf5')
model.fit(x_train, y_train, epochs=16, batch_size=16,
          validation_split=0.3, callbacks=[es,mcp])

model.save('./_save/keras30_2_save_model.h5')

#4. 평가, 예측
print ('====================== 1. 기본출력 ========================')
loss = model.evaluate(x_test, y_test)
print('loss:', loss)

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2 스코어:', r2)



'''
scaler = MaxAbsScaler()         
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
'''

'''
평가지표 = acc 
0.98 이상
train과 test과 분리되어 
        (평가용으로만)
        
        ====================== 1. 기본출력 ========================
313/313 [==============================] - 1s 3ms/step - loss: 0.0692 - accuracy: 0.9819
loss: [0.06917189061641693, 0.9818999767303467]
'''
################### CNN 주요 내용 정리 #########################
'''
model.add(Conv2D(a, kernel_size = (b,c), input_shape = (q,w,e))) 
1) a = 출력 채널 

2) b,c = 필터, 커넬 사이즈와 같은 말이며, 이미지의 특징을 찾아내기 위한
공용 파라미터이다. 

3) e : 입력 채널, RGB 
- 이미지 픽셀 하나하나는 실수이며, 컬러사진을 표현하기 위해서는 RGB 3개의
실수로 표현해야 한다. 

추가 

'''