from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.model_selection import cross_validate, KFold
from sklearn.utils import shuffle
import pandas as pd
from sklearn.metrics import accuracy_score

#1. 데이터
fold_iris = load_iris()
features = fold_iris.data   # 독립 변수
label = fold_iris.target
fold_df_clf = DecisionTreeClassifier()

kfold = KFold(n_splits=5)
cv_accuracy = []

n_iter = 0 
for train_idx, test_idx in kfold.split(features):
    X_train, X_test = features[train_idx], features[test_idx]
    y_train, y_test = label[train_idx],label[test_idx]
    fold_df_clf.fit(X_train, y_train)
    fold_pred = fold_df_clf.predict(X_test)

    n_iter += 1
    accuracy = np.round(accuracy_score(y_test,fold_pred),4)
    print('\n{} 교차검증정확도:{}, 학습데이터 크기: {}, 검증데이터 크기 : {}'.format(n_iter, accuracy, X_train.shape[0], X_test.shape[0]))
    cv_accuracy.append(accuracy)
print('\n')
print('\n 평균검증 정확도 :', np.mean(cv_accuracy)) 



print(X_train.shape, y_train.shape)   # (120, 4) (120,)


# #2. 모델구성
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense

# model = Sequential()
# model.add(Dense(128, activation= 'linear', input_dim=4))
# model.add(Dense(64, activation= 'sigmoid')) 
# model.add(Dense(32, activation= 'linear'))
# model.add(Dense(18, activation= 'linear'))
# model.add(Dense(9, activation= 'sigmoid'))
# model.add(Dense(3, activation = 'softmax')) # 결과치가 0과1이나오게 하기 위해 

# #3. 컴파일, 훈련
# model.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])
 
# from tensorflow.keras.callbacks import EarlyStopping
# es = EarlyStopping(monitor='val_loss', patience=10, mode='auto',
#                    verbose=1, restore_best_weights=False) # restore_best_weights=True : 최종값 이전에 가장 좋았던 값 도출함

# model.fit(X_train, y_train, epochs=50, batch_size=2,
#           validation_split=0.3)

# #4. 평가, 예측
# loss = model.evaluate(X_test, y_test)
# print('loss :', loss[0])
# print('accuracy :', loss[1])

# results = model.predict(X_test[:7])
# print(results)