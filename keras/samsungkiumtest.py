import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense,LSTM, Input

path = '../_data/juga predict/'  
SD = pd.read_csv(path + '삼성전자.csv',thousands=',')
SD = SD.drop(range(893, 1120), axis=0)
x1 = SD.drop(['일자','종가',"Unnamed: 6",'전일비','등락률','금액(백만)','신용비','개인','기관','외인(수량)','외국계','프로그램', '외인비'], axis=1) # axis=1 컬럼 삭제할 때 필요함
y = SD['종가']

KD = pd.read_csv(path + '키움증권.csv',thousands=',')
KD = KD.drop(range(892, 1059), axis=0)
x2 = KD.drop(['일자',"Unnamed: 6",'전일비','등락률','금액(백만)','신용비','개인','기관','외인(수량)','외국계','프로그램', '외인비'], axis=1)

print(x1.shape,y.shape,x2.shape)
# print(x2[893:1056])



size = 15

def split_x(data, size):
    aaa = []
    for i in range(len(data) - size +1): 
        subset = data[i : (i+size)]       
        aaa.append(subset)                  
    return np.array(aaa)
x1 = split_x(x1,size)
y = split_x(y,size)
x2 = split_x(x2,size)

print(x1.shape,y.shape,x2.shape) #(879, 15, 4) (879, 15) (879, 15, 5)

# print(x.columns, x.shape)  # (1060, 13)
  # (1060, 4) (1060,)

# x = x.to_numpy()

# x = x.head(10)
# y = y.head(20)

x1_train, x1_test, x2_train, x2_test,y_train, y_test = train_test_split(x1,x2,y,
        train_size =0.8, shuffle=True, random_state = 42)


print(x1_train.shape,y_train.shape,x2_train.shape,x1_test.shape)  # (874, 20, 4) (874, 20)


#2-1 모델1
input1 = Input(shape=(4,))   # (70,2) 
dense1 = Dense(5, activation='relu', name = 'dense1')(input1)
dense2 = Dense(7, activation='relu', name = 'dense2')(dense1)
dense3 = Dense(7, activation='relu', name = 'dense3')(dense2)
output1 = Dense(7, activation='relu', name = 'output1')(dense3)

#2-2 모델2
input2 = Input(shape=(5,))   # (70,3) 
dense11 = Dense(10, activation='relu', name = 'dense11')(input2)
dense12 = Dense(10, activation='relu', name = 'dense12')(dense11)
dense13 = Dense(10, activation='relu', name = 'dense13')(dense12)
dense14 = Dense(10, activation='relu', name = 'dense14')(dense13)
output2 = Dense(5, activation='relu', name = 'output2')(dense14)

from tensorflow.keras.layers import Concatenate, concatenate  
merge1 = Concatenate(axis=1)([output1,output2])       # Concatenate
merge2 = Dense(10, activation='relu')(merge1)
merge3 = Dense(7)(merge2)
last_output = Dense(1)(merge3)

model = Model(inputs = [input1,input2], outputs = last_output)

model.summary()

#3. 컴파일, 훈련
model.compile(loss='mae', optimizer = 'adam')
 
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', patience=20, mode='auto',
                   verbose=1, restore_best_weights=False)
mcp = ModelCheckpoint (monitor = 'val_loss', mode = 'min', verbose = 1, save_best_only=True,
                       filepath = '../Study/_ModelCheckPoint/kovt.hdf5')
model.fit([x1_train,x2_train], y_train, epochs=100, batch_size=2,
          validation_split=0.3)

model.save('../_test/_save/kovt.h5')

#4. 평가, 예측
loss = model.evaluate([x1_test,x2_test], y_test)
print('loss:', loss)

y_predict = model.predict([x1_test,x2_test])
print(y_predict)
