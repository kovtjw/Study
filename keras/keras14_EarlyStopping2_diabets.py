from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import r2_score
from sklearn.datasets import load_boston, load_diabetes
import time

#1.데이터
dataset = load_diabetes()
x = dataset.data
y = dataset.target

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, 
         train_size = 0.8, shuffle = True, random_state = 66) 


model = Sequential() 
model.add(Dense(deep_len[100], input_dim = 10)) 
model.add(Dense(deep_len[55])) 
model.add(Dense(deep_len[110]))
model.add(Dense(deep_len[30])) 
model.add(Dense(deep_len[14])) 
model.add(Dense(1)) 

#3. 컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping
patience_num = 50
es = EarlyStopping(monitor='val_loss', patience=patience_num, mode = 'min', verbose=1)
epoch = 10000
model.compile(loss = 'mse', optimizer = 'adam')
hist = model.fit(x_train, y_train, epochs = epoch, batch_size =1,validation_split=0.2,callbacks=[es])


end = time.time() - start
print('시간 : ', round(end,2) ,'초')

#4 평가예측
loss = model.evaluate(x_test,y_test)
print("loss : ",loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test,y_predict)
print("r2 : ",r2)

print("epochs :",epoch)


