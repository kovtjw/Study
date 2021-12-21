import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout, Conv1D, Flatten, Input
from sklearn.model_selection import train_test_split


#1. 데이터

path = '../_data/juga predict/'  
SD = pd.read_csv(path +"삼성전자.csv", thousands=',')
SD = SD.drop(range(61, 1120), axis=0)                                  # 액면분할 (893, 1120)
KD = pd.read_csv(path + "키움증권.csv", thousands=',')
KD = KD.drop(range(61, 1060), axis=0)
# ki = ki.drop(range(893, 1120), axis=0)

SD = SD.loc[::-1].reset_index(drop=True)
KD = KD.loc[::-1].reset_index(drop=True)
# y_ss = ss['종가']
# y_ki = ki['종가']


sx = SD.drop(['일자', '시가', '고가', '저가', '종가','전일비', 'Unnamed: 6', '등락률', '신용비', '외인비'], axis =1)
sx = np.array(sx)
kx = KD.drop(['일자', '시가', '고가', '저가', '종가', '전일비', 'Unnamed: 6', '등락률', '신용비', '외인비'], axis =1)
kx = np.array(sx)
xx1 = SD.drop(['일자', '시가', '고가', '저가', '종가','전일비', 'Unnamed: 6', '등락률', '신용비', '외인비', '거래량'], axis =1)
xx1 = np.array(xx1)
xx2 = KD.drop(['일자', '시가', '고가', '저가', '종가', '전일비', 'Unnamed: 6', '등락률', '신용비', '외인비', '거래량'], axis =1)
xx2 = np.array(xx2)



def split_xy3(dataset, time_steps, y_column):                     # size : 몇개로 나눌 것인가
    x, y = list(), list()
    for i in range(len(dataset)):
        x_end_number = i + time_steps
        y_end_number = x_end_number + y_column - 1
        
        if y_end_number > len(dataset):
            break
        tmp_x = dataset[i:x_end_number, 1:]                       # [a,b]  a= 데이터의 위치, b = 칼럼의 위치
        tmp_y = dataset[x_end_number - 1:y_end_number, 0]
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)

x1, y1 = split_xy3(sx, 5, 3)
x2, y2 = split_xy3(kx, 5, 3)

print(x1.shape, y1.shape)     # (194, 5, 5) (194, 3)
print(x2.shape, y2.shape)     # (194, 5, 5) (194, 3)

def split_x(dataset, size):                     # size : 몇개로 나눌 것인가
    aaa = []
    for i in range(len(dataset) - size + 1):
        subset = dataset[i : (i + size)]
        aaa.append(subset)
    return np.array(aaa)

pred1 = split_x(xx1, 5)
pred2 = split_x(xx2, 5)


x1_train, x1_test, x2_train, x2_test, y1_train, y1_test, y2_train, y2_test = train_test_split(
    x1, x2, y1, y2, train_size=0.7, shuffle=True, random_state=42)


# #2-1 모델1
# input1 = Input((5,6))
# dense1 = LSTM(16, activation='relu')(input1)
# dense2 = Dense(64, activation='relu')(dense1)
# dense3 = Dense(32, activation='relu')(dense2)
# dense4 = Dense(16, activation='relu')(dense3)
# output1 = Dense(8, activation='relu')(dense4)


# #2-2 모델2
# input2 = Input((5,6))
# dense11 = LSTM(32, activation='relu')(input2)
# dense12 = Dense(64, activation='relu')(dense11)
# dense13 = Dense(32, activation='relu')(dense12)
# dense14 = Dense(16, activation='relu')(dense13)
# output2 = Dense(8, activation='relu')(dense14)

# from tensorflow.keras.layers import concatenate, Concatenate
# # merge1 = concatenate([output1, output2])            # (None, 12)
# # print(merge1.shape)
# merge1 = Concatenate()([output1, output2])            # (None, 12)
# # print(merge1.shape)

# #2-3 output 모델1
# output21 = Dense(16)(merge1)
# output22 = Dense(8, activation='relu')(output21)
# output23 = Dense(2, activation='relu')(output22)
# last_output1 = Dense(1)(output23)

# #2-4 output 모델2
# output31 = Dense(16)(merge1)
# output32 = Dense(8, activation='relu')(output31)
# output33 = Dense(2, activation='relu')(output32)
# last_output2 = Dense(1)(output33)

# model = Model(inputs=[input1, input2], outputs=[last_output1,last_output2])

# model.summary()

# #3. 컴파일, 훈련
# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# model.compile(loss='mae', optimizer = 'adam')
 
# es = EarlyStopping(monitor='val_loss', patience=30, mode='auto',
#                    verbose=1, restore_best_weights=True)
# mcp = ModelCheckpoint (monitor = 'val_loss', mode = 'min', verbose = 1, save_best_only=True,
#                        filepath = '../Study/_ModelCheckPoint/kovt.hdf5')
# hist = model.fit([x1_train, x2_train], [y1_train, y2_train], epochs=500, batch_size=1, validation_split=0.3, callbacks=[mcp])

model = load_model ('../_test/_save/kovt4.h5')
#4. 평가, 예측
loss = model.evaluate ([x1_test, x2_test], [y1_test,y2_test], batch_size=1)
print('loss :', loss)
# pred1 = np.array(x_ss[-5:,1:])
result1, result2 = model.predict([pred1, pred2])

model.save('../_test/_save/kovt5.h5'.format(result1[-1][-1],result2[-1][-1]))

print('삼성전자 12/21 거래량 : ', result1[-1][-1],'주')
print('키움증권 12/21 거래량 : ', result2[-1][-1],'주')


'''
12/20 거래량 
삼성전자 거래량 예측값 :  [11,170,845]
키움증권 거래량 예측값 :  [49,658]
삼성전자 12/21 거래량 :  9532456.0 주
키움증권 12/21 거래량 :  43826.39 주
'''