from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error 
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, LabelEncoder
import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras.metrics import accuracy

#1. 데이터

path = '../_data/dacon/wine/'  # 점하나는 지금 작업하고 있는 위치, 점 두개는 이전 폴더
train = pd.read_csv(path+'train.csv')
test_file = pd.read_csv(path+'test.csv')
submit_file = pd.read_csv(path + 'sample_submission.csv')

x = train.drop(['quality'],axis=1)
y = train['quality']

le = LabelEncoder()   # string으로 되어있는 문자값을 숫자값으로 바꾸어 주는 것, (남자, 여자, 외계인, 원숭이) > (0,1,2,3)
# le.fit(x['type'])
le.fit(train.type)
x['type'] = le.transform(x['type'])

# le.fit(test_file['type'])
# test_file['type'] = le.transform(test_file['type'])
y = to_categorical(y)  # 0부터 순차적으로 채워준다. > 99와 100밖에 없을 때에는 0부터 100까지 101개의 카테고리컬이 생성이 된다.
# print(y.shape)      # (3231, 9)

x_train, x_test, y_train, y_test = train_test_split(x,y,
         train_size =0.9, shuffle=True, random_state = 42)  
print(x.type.value_counts())  # 1    2453  / 0     778 >> .pandas에서만 가능함, 수치 데이터들은 numpy이다. 
# x['type'] = x.type 표현 방식이 같다.


scaler = MaxAbsScaler()         
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델구성
model = Sequential()
model.add(Dense(100, input_dim=13)) 
model.add(Dense(80, activation='relu'))
model.add(Dense(130, activation='sigmoid'))
model.add(Dense(80, activation='relu'))
model.add(Dense(5))
model.add(Dense(9,activation='softmax'))   # 9개의 컬럼으로 출력이 되는 형태 

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])
 
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=100, mode='auto',
                   verbose=1, restore_best_weights=True) # restore_best_weights=True : 최종값 이전에 가장 좋았던 값 도출함

model.fit(x_train, y_train, epochs=1000, batch_size=32,
          validation_split=0.3, callbacks=[es])

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss :', loss[0])
print('accuracy :', loss[1])

le.fit(test_file.type)
test_file_type = le.transform(test_file['type'])
test_file['type'] = test_file_type

# print(test_file)

scaler.transform(test_file)

# ##################### 제출용 제작 ####################
results = model.predict(test_file)
result_recover = np.argmax(results, axis =1).reshape(-1,1)
# print(results[:5])
submit_file['quality'] = result_recover
# print(np.unique(result_recover[:5]))
# submit_file.to_csv(path+"wineryy.csv", index = False)
# submit_file ['quality'] = results

# print(submit_file[:10])

# # submit_file.to_csv(path + 'Suu.csv', index=False) # to_csv하면 자동으로 인덱스가 생기게 된다. > 없어져야 함
submit_file.to_csv(path+"winerryy.csv", index = False)
# print(result_recover)
acc= str(round(loss[1]*100,4))
model.save(f"./_save/keras24_dacon_save_model_{acc}.h5")
