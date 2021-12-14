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
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score


# 1. 데이터

path = '../_data/dacon/cardiovascular disease/'
train = pd.read_csv(path+'train.csv')
test_file = pd.read_csv(path+'test.csv')
submit_file = pd.read_csv(path + 'sample_submission.csv')


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
         train_size =0.7, shuffle=True, random_state = 2)

# print(y_train.shape)  # (135, 14)
# print(y_train.shape)
# print(x_test.shape)  # (16, 14)
# print(y_test.shape)

# scaler = RobustScaler()         
# scaler.fit(x_train)
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

#2. 모델구성
def mlp_model():
    model = Sequential()
    model.add(Dense(100, input_dim=13)) 
    model.add(Dropout(0.3)) 
    model.add(Dense(130, activation='relu'))
    model.add(Dropout(0.5)) 
    model.add(Dense(100, activation='relu'))
    model.add(Dense(80, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))


#3. 컴파일, 훈련
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])
    return model
model = mlp_model()

# 서로 다른 모델을 3개 만들어 합친다
model1 = KerasClassifier(build_fn = mlp_model, epochs = 100, verbose = 1)
model1._estimator_type="classifier" 
model2 = KerasClassifier(build_fn = mlp_model, epochs = 200, verbose = 1)
model2._estimator_type="classifier"
model3 = KerasClassifier(build_fn = mlp_model, epochs = 300, verbose = 1)
model3._estimator_type="classifier"
model4 = KerasClassifier(build_fn = mlp_model, epochs = 400, verbose = 1)
model4._estimator_type="classifier"
model5 = KerasClassifier(build_fn = mlp_model, epochs = 500, verbose = 1)
model5._estimator_type="classifier"


ensemble_clf = VotingClassifier(estimators = [('model1', model1), 
                                              ('model2', model2), 
                                              ('model3', model3), 
                                              ('model4', model4), 
                                              ('model5', model5)], voting = 'soft')
ensemble_clf.fit(x_train, y_train)

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', patience=50, mode='min', verbose=1) 


model.fit(x_train, y_train, epochs=1000, batch_size=2,
          validation_split=0.111, callbacks=[es])
# 서로 다른 모델을 3개 만들어 합친다


#4. 평가, 예측
print ('====================== 1. 기본출력 ========================')
loss = model.evaluate(x_test, y_test)
y_pred = model.predict(x_test)
y_pred = y_pred.round(0).astype(int)
f1 = f1_score(y_pred, y_test)
print('f1 스코어 :',f1)

# print('loss:', loss)

results = model.predict(test_file)
results = results.round(0).astype(int)

# print('f1 스코어 :',f1)
# print(f1.shape)

##################### 제출용 제작 ####################

# result_recover = np.argmax(results, axis =1).reshape(-1,1)
print(results[:5])
submit_file['target'] = results
submit_file.to_csv(path+"likedog4.csv", index = False)



# f1-score를 사용하게된 이유

