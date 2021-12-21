import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout, Conv1D, Flatten, Input
from sklearn.model_selection import train_test_split


#1. 데이터

path = '../_data/juga predict/'  
SD = pd.read_csv(path +"삼성전자.csv", thousands=',')
SD = SD.drop(range(41, 1120), axis=0)                                  # 액면분할 (893, 1120)
KD = pd.read_csv(path + "키움증권.csv", thousands=',')
KD = KD.drop(range(41, 1060), axis=0)

SD = SD.loc[::-1].reset_index(drop=True)
KD = KD.loc[::-1].reset_index(drop=True)

# print(SD,KD)

sx = SD.drop(['일자', '고가', '저가', '전일비', 'Unnamed: 6', '등락률', '거래량', '금액(백만)', '신용비','개인', '기관', '외인(수량)','외국계','프로그램','외인비'], axis =1)
sx = np.array(sx)
kx = KD.drop(['일자', '고가', '저가', '전일비', 'Unnamed: 6', '등락률', '거래량', '금액(백만)', '신용비','개인', '기관', '외인(수량)','외국계','프로그램','외인비'], axis =1)
kx = np.array(sx)
xx1 = SD.drop(['일자', '고가', '저가', '전일비', 'Unnamed: 6', '등락률', '거래량', '금액(백만)', '신용비','개인', '기관', '외인(수량)','외국계','프로그램','외인비'], axis =1)
xx1 = np.array(xx1)
xx2 = KD.drop(['일자', '고가', '저가', '전일비', 'Unnamed: 6', '등락률', '거래량', '금액(백만)', '신용비','개인', '기관', '외인(수량)','외국계','프로그램','외인비'], axis =1)
xx2 = np.array(xx2)

print(sx.shape,kx.shape) # (21, 2) (21, 2)

def split_xy3(dataset, time_steps, y_column):                     # size : 몇개로 나눌 것인가
    x, y = list(), list()
    for i in range(len(dataset)):
        x_end_number = i + time_steps
        y_end_number = x_end_number + y_column 
        
        if y_end_number > len(dataset):
            break
        tmp_x = dataset[i:x_end_number, :]                       # [a,b]  a= 데이터의 위치, b = 칼럼의 위치
        tmp_y = dataset[x_end_number:y_end_number, 0]
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)

x1, y1 = split_xy3(sx, 3, 3)
x2, y2 = split_xy3(kx, 3, 3)

print('x값 : ',x1,'\n','y값 : ',y1)

print(x1.shape, y1.shape)     # (16, 3, 2) (16, 3)
print(x2.shape, y2.shape)     # (16, 3, 2) (16, 3)

# print(xx1[-3:,1])


x1_train, x1_test, x2_train, x2_test, y1_train, y1_test, y2_train, y2_test = train_test_split(
    x1, x2, y1, y2, train_size=0.7, shuffle=True, random_state=42)
print(x1_test.shape, x2_train.shape) # (8, 4, 2) (17, 4, 2)

#2-1 모델1
input1 = Input((3,2))
dense1 = LSTM(36, activation='relu')(input1)
dense2 = Dense(36, activation='relu')(dense1)
dense3 = Dense(18, activation='relu')(dense2)
dense4 = Dense(9, activation='relu')(dense3)
output1 = Dense(4, activation='relu')(dense4)

#2-2 모델2
input2 = Input((3,2))
dense11 = LSTM(36, activation='relu')(input2)
dense12 = Dense(36, activation='relu')(dense11)
dense13 = Dense(18, activation='relu')(dense12)
dense14 = Dense(9, activation='relu')(dense13)
output2 = Dense(4, activation='relu')(dense14)

from tensorflow.keras.layers import concatenate, Concatenate
# merge1 = concatenate([output1, output2]) 
# print(merge1.shape)
merge1 = Concatenate()([output1, output2]) 
print(merge1.shape) # (None, 16)

#2-3 output 모델1
output21 = Dense(16)(merge1)
output22 = Dense(8, activation='relu')(output21)
output23 = Dense(2, activation='relu')(output22)
last_output1 = Dense(3)(output23)

#2-4 output 모델2
output31 = Dense(16)(merge1)
output32 = Dense(8, activation='relu')(output31)
output33 = Dense(2, activation='relu')(output32)
last_output2 = Dense(3)(output33)

model = Model(inputs=[input1, input2], outputs=[last_output1,last_output2])



#3. 컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

model.compile(loss='mae', optimizer = 'adam')
 
es = EarlyStopping(monitor='val_loss', patience=20, mode='auto',
                   verbose=1, restore_best_weights=True)
mcp = ModelCheckpoint (monitor = 'val_loss', mode = 'min', verbose = 1, save_best_only=True,
                       filepath = '../Study/_ModelCheckPoint/kovt.hdf5')
hist = model.fit([x1_train, x2_train], [y1_train, y2_train], epochs=200, batch_size=2, validation_split=0.3, callbacks=[mcp])


# model = load_model ('../_test/_save/kovt6.h5')
#4. 평가, 예측
loss = model.evaluate ([x1_test, x2_test], [y1_test,y2_test], batch_size=1)
print('loss :', loss)

xx1 = xx1[-3:,:].reshape(1, 3, 2)
xx2 = xx2[-3:,:].reshape(1, 3, 2)

# pred1,pred2 = np.array()
result1,result2= model.predict([xx1,xx2])
print(result1,result2)
# model.save('../_test/_save/kovt5.h5'.format(result1[-1][-1],result2[-1][-1]))

print('삼성전자 12/21 시가 : ', result1[-1][-1],'원')
print('키움증권 12/21 시가 : ', result2[-1][-1],'원')
 