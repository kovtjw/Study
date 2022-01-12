from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow.keras.layers import Dense, LSTM
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_validate, KFold
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from matplotlib import font_manager, rc
font_path = "C:/Windows/Fonts/malgunbd.ttf"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)
import seaborn as sns
import pandas as pd
import graphviz 
import multiprocessing
import matplotlib.pyplot as plt
plt.style.use(['seaborn-whitegrid'])
from sklearn import tree
from sklearn.pipeline import make_pipeline

path = '../_data/project data/' 
gwangju = pd.read_csv(path +"data_광주.csv")
x = gwangju.drop(['일자','가격'], axis = 1)
y = gwangju['가격']


x_train,x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

fold_df_clf = RandomForestRegressor()
kfold = KFold(n_splits=5)
cv_accuracy = []

n_iter = 0 
for train_idx, test_idx in kfold.split(x):
    x_train, x_test = x_train, x_test
    y_train, y_test = y_train, y_test
    fold_df_clf.fit(x_train, y_train)
    fold_pred = fold_df_clf.predict(x_test)

    n_iter += 1
    accuracy = np.round(accuracy_score(y_test,fold_pred),4)
    print('\n{} 교차검증정확도:{}, 학습데이터 크기: {}, 검증데이터 크기 : {}'.format(n_iter, accuracy, x_train.shape[0], x_test.shape[0]))
    cv_accuracy.append(accuracy)
print('\n')
print('\n 평균검증 정확도 :', np.mean(cv_accuracy)) 

#2. 모델구성
model = Sequential()
model.add(Dense(64, input_dim=4)) 
model.add(Dense(128, activation='relu'))      
model.add(Dense(64, activation='relu'))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(1))

#3. 컴파일, 훈련 

model.compile(loss = 'mse', optimizer = 'adam')   # optimizer는 loss값을 최적화 한다.
model.fit(x_train, y_train, epochs = 100)
RandomForestClassifier.fit(n_estimators=100, learning_rate=0.08, gamma=0, 
                     subsampel=0.75, colsample_bytree=1, max_depth=7)

#4. 평가, 예측 
loss = model.evaluate(x_test, y_test)
print('loss:', loss)

y_predict = model.predict(x_test)

dot_data = tree.export_graphviz(decision_tree=RandomForestClassifier(),
                                feature_names= x,
                                class_names = y,
                                filled=True, rounded=True,
                                special_characters = True)

graph = graphviz.Source(dot_data)
print(graph)