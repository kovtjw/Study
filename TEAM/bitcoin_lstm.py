import tensorflow as tf, pandas as pd, numpy as np
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense,Bidirectional,GRU, Dropout
from keras.callbacks import EarlyStopping , ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

def split_xy_yr(dataset, time_steps, y_time_steps):                
    x,y = list(), list()                                    

    for i in range(len(dataset)):                           
        x_end_number = i + time_steps                        
        y_end_number = x_end_number + y_time_steps            

        if y_end_number > len(dataset):                        
            break

        tmp_x = dataset[i:x_end_number,0:-1]
        tmp_y = dataset[x_end_number:y_end_number,-1] # , -1     
        x.append(tmp_x)
        y.append(tmp_y)
        
    return np.array(x),np.array(y)

path = os.path.dirname(os.path.realpath(__file__))

bitcoin = pd.read_csv(path + '/BTC_USD.csv',header=0).iloc[:370,:]\
        .sort_values(['날짜'],ascending=True,ignore_index=True)\
        .rename(columns={'오픈':'시가','변동 %':'타겟'})\
        [['날짜','시가','고가','저가','종가','거래량','타겟']]

price_list = ['시가','고가','저가','종가']
    
for p in price_list:
    for i,price in enumerate(bitcoin[p]):
        price = float(price.replace(',',''))
        bitcoin[p][i] = price

for i,deal in enumerate(bitcoin['거래량']):
    deal = deal.replace('K','000').replace('.','')
    bitcoin['거래량'][i] = deal
    
for i,target in enumerate(bitcoin['타겟']):
    target = float(target.replace('%',''))
    if target > 0: bitcoin['타겟'][i] = 1
    elif target < 0: bitcoin['타겟'][i] = 0

data = np.array(bitcoin[['시가','종가','타겟']])

x,y = split_xy_yr(data,3,1)
split_time = 50

x_train = x[:split_time]
x_test = x[split_time:]
y_train = y[:split_time]
y_test = y[split_time:]

x_train=np.asarray(x_train).astype(np.float64)
x_test=np.asarray(x_test).astype(np.float64)
y_train=np.asarray(y_train).astype(np.float64)
y_test=np.asarray(y_test).astype(np.float64)

model = Sequential()
model.add(LSTM(100,input_shape=(3,2)))
model.add(LSTM(128, activation='relu'))
model.add(Dropout(0.2))
model.add(LSTM(64, activation='relu'))
model.add(LSTM(2, activation='relu'))
model.add(Dense(2,activation='softmax'))

model.summary()
# exit()
optimizer = Adam(learning_rate=0.0001)  # 1e-4     
lr=ReduceLROnPlateau(monitor= "val_acc", patience = 60, mode='max',factor = 0.1, min_lr=0.000001,verbose=False)
es = EarlyStopping(monitor ="val_acc", patience=15, mode='max',verbose=1,restore_best_weights=True)

model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['acc'])    

model.fit(x_train,y_train,batch_size=2,epochs=1000,validation_data=(x_test,y_test),callbacks=[lr,es], verbose=1)

y_pred = model.predict(x_test)
y_pred_int  = np.argmax(y_pred,axis=1)
print(y_pred[-1:])
print(y_pred_int[-1:])
