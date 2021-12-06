from tensorflow.keras import callbacks
from tensorflow.keras.models import Sequential ,load_model
from tensorflow.keras.layers import Dense
import numpy as np 
from sklearn.model_selection import train_test_split
import time
from sklearn.datasets import load_boston
from tensorflow.python.keras.saving.save import load_model

#1. 데이터
datasets = load_boston()

x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x,y,
        train_size =0.8, shuffle=True, random_state = 42)

# # 2. 모델구성
# model = Sequential()
# model.add(Dense(100, input_dim=13)) 
# model.add(Dense(80))
# model.add(Dense(130))
# model.add(Dense(80))
# model.add(Dense(5))
# model.add(Dense(1))


#3. 컴파일, 훈련
# model.compile(loss='mse', optimizer='adam')
# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# es = EarlyStopping(monitor='val_loss', patience = 10, mode = 'min', verbose = 1, 
#                    restore_best_weights=True)  # 여기서 'verbose'를 하면, Restoring model weights from the end of the best epoch.를 보여준다. 
# mcp = ModelCheckpoint(monitor = 'val_loss', mode = 'auto', verbose = 1, save_best_only=True,
#                       filepath = './_ModelCheckPoint/keras26_1_MCP.hdf5')

# start = time.time()
# hist = model.fit(x_train, y_train, epochs=100, batch_size=8,
#           validation_split=0.3, callbacks = [es,mcp])
# end = time.time() - start

# print(hist.history['loss'])
# print('===================================')
# print(hist.history['val_loss'])
# print('===================================')

# print('걸린시간:', round(end,3), '초')

# model.save('./_save/keras26_1_save_model.h5')  ##
# model.save_weights('./_save/keras25_1_save_weights.h5')
# model.load_weigths('./_save/keras25_3_save_model.h5') 
model = load_model('./_ModelCheckPoint/28-41.59.h5')
# model = load_model('./_save/keras26_1_save_model.h5')


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss:', loss)

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2 스코어:', r2)

'''
<mcp>
loss: 36.27262878417969
r2 스코어: 0.44059372504553984
<save_model>
loss: 36.27262878417969
r2 스코어: 0.44059372504553984

restore_best_weights=True 제외 했을 때
<mcp>
loss: 41.41362762451172
r2 스코어: 0.3613078079847448
<save_model>
'''
