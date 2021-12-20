import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense,LSTM, Input
from tensorflow.keras.layers import Concatenate, concatenate  
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


path = '../_data/juga predict/'  
SD = pd.read_csv(path + '삼성전자.csv',thousands=',')
KD = pd.read_csv(path + '키움증권.csv',thousands=',')

SD = SD.drop(range(7, 1120), axis=0)
KD = KD.drop(range(7, 1060), axis=0)

x1 = SD.drop(['일자','고가','저가',"Unnamed: 6",'전일비','등락률','거래량','금액(백만)','신용비','개인','기관','외인(수량)','외국계','프로그램', '외인비'], axis=1) # axis=1 컬럼 삭제할 때 필요함
x1 = np.array(x1)
x2 = KD.drop(['일자','고가','저가',"Unnamed: 6",'전일비','등락률','거래량','금액(백만)','신용비','개인','기관','외인(수량)','외국계','프로그램', '외인비'], axis=1)
x2 = np.array(x2)

def split_xy(dataset, time_steps, y_column):
    x, y =list(),list()
    for i in range(len(dataset)): 
        x_end_number = i + time_steps       
        y_end_number = x_end_number + y_column - 1                  
    
        if y_end_number > len(dataset):
            break
        tmp_x = dataset[i:x_end_number, :-1]
        tmp_y = dataset[x_end_number-1:y_end_number, -1]
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)
    
x1, y1 = split_xy(x1,4,2)
x2, y2 = split_xy(x2,4,2)

x1_train, x1_test, y1_train, y1_test = train_test_split(x1,y1,
        train_size =0.7, shuffle= True, random_state = 42)
x2_train, x2_test,y2_train,y2_test = train_test_split(x2,y2,
        train_size = 0.7, shuffle = True, random_state = 42)


#2-1 모델1
input1 = Input(shape=(4,1))   # (70,2) 
dense1 = LSTM(32, activation='relu')(input1)
dense2 = Dense(64, activation='relu')(dense1)
dense3 = Dense(32, activation='relu')(dense2)
dense4 = Dense(16, activation='relu')(dense3)
output1 = Dense(8, activation='relu')(dense4)

#2-2 모델2
input2 = Input(shape=(4,1))   # (70,3) 
dense11 = LSTM(32, activation='relu')(input2)
dense12 = Dense(64, activation='relu')(dense11)
dense13 = Dense(32, activation='relu')(dense12)
dense14 = Dense(16, activation='relu')(dense13)
output2 = Dense(8, activation='relu')(dense14)

merge1 = Concatenate()([output1,output2]) 

# 2-3 output모델1
output21 = Dense(16)(merge1)
output22 = Dense(8, activation='relu')(output21)
output23 = Dense(2, activation='relu')(output22)
last_output1 = Dense(1)(output23)  # y의 열의 갯수

# 2-4 output모델2
output31 = Dense(16)(merge1)
output32 = Dense(8, activation='relu')(output31)
output33 = Dense(2, activation='relu')(output32)
last_output2 = Dense(1)(output33)

model = Model(inputs = [input1,input2], outputs= ([last_output1,last_output2]))


#3. 컴파일, 훈련
model.compile(loss='mae', optimizer = 'adam')
 
es = EarlyStopping(monitor='val_loss', patience=10, mode='auto',
                   verbose=1, restore_best_weights=True)
mcp = ModelCheckpoint (monitor = 'val_loss', mode = 'min', verbose = 1, save_best_only=True,
                       filepath = '../Study/_ModelCheckPoint/kovt.hdf5')
model.fit([x1_train,x2_train], [y1_train,y2_train], epochs=300, batch_size=2,
          validation_split=0.3)

# model = load_model ('../_test/_save/kovt3.h5')

model.save('../_test/_save/kovt4.h5')

#4. 평가, 예측
loss = model.evaluate ([x1_test, x2_test], [y1_test,y2_test], batch_size=1)
print('loss :', loss) #loss :
y1_pred, y2_pred = model.predict([x1, x2])
print('삼성예측값 : ', y1_pred[-1])
print('키움예측값 : ', y2_pred[-1])

'''
삼성예측값 :  [77218.99]
키움예측값 :  [107550.266]
'''