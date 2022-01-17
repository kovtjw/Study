import numpy as np
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split, KFold
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, r2_score

#1. 데이터
datasets = load_iris()
x = datasets.data
y = datasets.target


x_train, x_test, y_train, y_test = train_test_split(x,y,
        shuffle= True, random_state=66, train_size = 0.8)
n_splits = 5
kfold = KFold(n_splits = n_splits, shuffle = True, random_state=100)

parameters = [
    {'C':[1,10,100,1000], 'kernel':['linear'], 'degree' : [3,4,5]},  # 12개
    {'C':[1, 10 ,100], 'kernel':['rbf'], 'gamma':[0.001, 0.0001]},   # 6개
    {'C':[1, 10, 100, 1000], 'kernel': ['sigmoid'], 
     'gamma':[0.01, 0.001, 0.0001], 'degree': [3,4]}                 # 24개
]                                                                    # 총 42번(더해준다) 


#2. 모델 구성
model = GridSearchCV(SVC(), parameters, cv = kfold, verbose = 3)   # GridSearchCV(모델, 파라미터, cv = 크로스 발리데이션)
# model = SVC(c = 1, kernel ='linear', degree = 3)

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측

print('최적의 매개 변수 :', model.best_estimator_)
print('최적의 파라미터 : ', model.best_params_)


print('best_score_ :', model.best_score_)

aaa = model.score(x_test, y_test)      # evaluate 개념
print('model.score :', model.score(x_test, y_test))

y_pred = model.predict(x_test) 
print('accuracy_score :', accuracy_score(y_test, y_pred))


'''
최적의 매개 변수 : SVC(C=1, kernel='linear')
최적의 파라미터 :  {'C': 1, 'degree': 3, 'kernel': 'linear'}
best_score_ : 0.9666666666666666 >>> train에서 가장 좋은 값
model.score : 1.0                >>> test(or predict)에서 가장 좋은 값
accuracy_score : 1.0
'''