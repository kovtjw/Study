from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import numpy as np
import pandas as pd
import time
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, LabelEncoder, OneHotEncoder
from pandas import get_dummies
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier


#1 데이터
path = "../_data/dacon/wine/"  
train = pd.read_csv(path +"train.csv")
test_file = pd.read_csv(path + "test.csv") 

submission = pd.read_csv(path+"sample_Submission.csv") 
y = train['quality']
x = train.drop(['id', 'quality'], axis =1)


le = LabelEncoder()                 
label = x['type']
le.fit(label)
x['type'] = le.transform(label)

print(x)                          # type column의 white, red를 0,1로 변환
print(x.shape)                    # (3231, 12)


test_file = test_file.drop(['id'], axis=1)
label2 = test_file['type']
le.fit(label2)
test_file['type'] = le.transform(label2)

y = train['quality']
y = get_dummies(y)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.9, shuffle = True, random_state = 66)

scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = RobustScaler()
# scaler = MaxAbsScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
test_file = scaler.transform(test_file)


model = Sequential()
model.add(Dense(100, input_dim=x.shape[1], activation='relu'))
model.add(Dense(80, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(8, activation='relu'))
model.add(Dropout(rate = 0.2))
model.add(Dense(5, activation='softmax'))

#3. 컴파일, 훈련

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics=['accuracy']) # metrics=['accuracy'] 영향을 미치지 않는다
from tensorflow.keras.callbacks import EarlyStopping
patience_num = 50
es = EarlyStopping(monitor='val_loss', patience=patience_num, mode = 'auto', verbose=1, restore_best_weights=True)
model.fit(x_train, y_train, epochs = 1000, batch_size =64, validation_split=0.1, callbacks=[es])


#4 평가예측
loss = model.evaluate(x_test,y_test)
print("loss : ",loss[0])                      # List 형태로 제공된다
print("accuracy : ",loss[1])



############################### 제출용 ########################################
result = model.predict(test_file)
print(result[:5])
result_recover = np.argmax(result, axis=1).reshape(-1,1) + 4
print(result_recover[:5])
print(np.unique(result_recover))                           # value_counts = pandas에서만 먹힌다. 
submission['quality'] = result_recover

# print(submission[:10])
submission.to_csv(path + "wowe.csv", index = False)

print(result_recover)