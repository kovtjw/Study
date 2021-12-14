from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Input, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler,MaxAbsScaler, LabelEncoder
import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.metrics import f1_score
from tensorflow.python.keras.backend import reshape


# 1. 데이터

path = '../_data/dacon/cardiovascular disease/'
train = pd.read_csv(path+'train.csv')
test_file = pd.read_csv(path+'test.csv')
submit_file = pd.read_csv(path + 'sample_submission.csv')

print(train.describe)

# print(train.shape)  # (151, 15)
# print(test_file.shape)  # (152, 14)
# print(submit_file.shape) # (152, 2)

x = train.drop(['id','target'],axis = 1)
test_file = test_file.drop(['id'],axis = 1)
y = train['target']

# x['target'] = y
# print(x.corr())
# plt.figure(figsize=(10,10))
# sns.heatmap(data=x.corr(), square=True, annot=True, cbar=True)
# plt.show()

# print(x.shape)  # (151, 11)
# print(test_file.shape)  # (152, 10)
# print(submit_file.shape)

x_train, x_test, y_train, y_test = train_test_split(x,y,
         train_size =0.7, shuffle=True, random_state = 42)


print(x_train[:20])
print(y_train.shape)  # (135, 14)
# print(y_train.shape)
# print(x_test.shape)  # (16, 14)
# print(y_test.shape)

# scaler = MinMaxScaler()         
# # scaler.fit(x_train)
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)


#2. 모델구성
model = Sequential()
model.add(Dense(500, input_dim=13)) 
model.add(Dropout(0.2)) 
# model.add(Dense(32, activation='relu'))
model.add(Dense(300, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(200, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer = 'adam')

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', patience=50, mode='min', verbose=1) 


model.fit(x_train, y_train, epochs=2000, batch_size=10,
          validation_split=0.2, callbacks=[es])


#4. 평가, 예측
print ('====================== 1. 기본출력 ========================')
loss = model.evaluate(x_test, y_test)
y_pred = model.predict(x_test)
# y_pred = y_pred.reshape((31, ))
# y_pred = y_pred.round(0).astype(int)


# print(type(y_pred))

# print(y_pred.shape)
# print('loss:', loss)
y_pred = y_pred.reshape(y_pred.shape[0],).round(0).astype(int)
# print(y_pred)
f1 = f1_score(y_pred, y_test)
# print('f1 스코어 :',f1)

results = model.predict(test_file)
# results = results.reshape((152,))
# results = results.round(1).astype(int)
results= results.reshape(results.shape[0],).round(0).astype(int)
print(results)

print('f1 스코어 :',f1)


##################### 제출용 제작 ####################

# result_recover = np.argmax(results, axis =1).reshape(-1,1)

submit_file['target'] = results
submit_file.to_csv(path+"likedog4.csv", index = False)



# f1-score를 사용하게된 이유

