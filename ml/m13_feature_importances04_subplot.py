# subplot 이용해서 4개의 모델을 한 화면에 그래프로

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

#1. 데이터

datasets = load_iris()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x,y,
        train_size =0.8, shuffle=True, random_state = 100)

#2. 모델구성
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

#2. 모델
model1 = DecisionTreeClassifier()
model2 = RandomForestClassifier()
model3 = XGBClassifier()
model4 = GradientBoostingClassifier()

#3. 훈련
model1.fit(x_train,y_train)
model2.fit(x_train,y_train)
model3.fit(x_train,y_train)
model4.fit(x_train,y_train)


#4. 평가, 예측
result = model1.score(x_test, y_test)

from sklearn.metrics import accuracy_score
y_predict1 = model1.predict(x_test)
y_predict2 = model2.predict(x_test)
y_predict3 = model3.predict(x_test)
y_predict4 = model4.predict(x_test)

acc1 = accuracy_score(y_test, y_predict1)
acc2 = accuracy_score(y_test, y_predict2)
acc3 = accuracy_score(y_test, y_predict3)
acc4 = accuracy_score(y_test, y_predict4)

print('Decision_score :', acc1)
print('Random_score :', acc2)
print('XGBC_score :', acc3)
print('Gradient_score :', acc4)

import matplotlib.pyplot as plt
import numpy as np

def plot_feature_importance_dataset(model):
    n_features = datasets.data.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_,
             align = 'center')
    plt.yticks(np.arange(n_features), datasets.feature_names)
    plt.xlabel('Feature Importance')
    plt.ylabel('Features')
    plt.ylim(-1, n_features)
    
plt.figure(figsize=(20,20))
plt.subplot(3, 3, 1)
plot_feature_importance_dataset(model1)
plt.subplot(3, 3, 2)
plot_feature_importance_dataset(model2)
plt.subplot(3, 3, 3)
plot_feature_importance_dataset(model3)
plt.subplot(3, 3, 4)
plot_feature_importance_dataset(model4)
plt.show()

'''
plt.figure(figsize=(20,20))
plt.subplot(2,2,1)
plot_feature_importance_dataset(model1)
plt.subplot(2,2,2)
plot_feature_importance_dataset(model2)
plt.subplot(2,2,3)
plot_feature_importance_dataset(model3)
plt.subplot(2,2,4)
plot_feature_importance_dataset(model4)
plt.show()

Decision_score : 0.9666666666666667
Random_score : 0.9666666666666667
XGBC_score : 0.9666666666666667
Gradient_score : 0.9666666666666667
'''
