from bayes_opt import BayesianOptimization
import numpy as np
import pandas as pd
import time 
from datetime import datetime
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, PowerTransformer, QuantileTransformer, RobustScaler, StandardScaler
from xgboost import XGBRegressor
# loc(인덱스와 컬럼명), iloc(인덱스의 수치) 

############## 아웃 라이어 함수 ###############
def outliers(data_out):
    quartile_1, q2, quartile_3 = np.percentile(data_out,
                                               [25, 50, 75])
    print('1사분위 : ', quartile_1)
    print('q2 :', q2)
    print('3사분위 :', quartile_3)
    iqr = quartile_3 - quartile_1
    print('iqr :', iqr)
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)
    return np.where((data_out>upper_bound) | (data_out<lower_bound))
######################################################

############## NMAE 함수 ###############
def NMAE(true, pred):
    mae = np.mean(np.abs(true-pred))
    score = mae / np.mean(np.abs(true))
    return score
#########################################



# object는 최상위 >> string으로 생각해야 함 

path = '../_data/dacon/housing/'
datasets = pd.read_csv(path + 'train.csv', index_col= 0, header=0)
test_sets = pd.read_csv(path + 'test.csv', index_col= 0, header=0)
submit_sets = pd.read_csv(path + 'sample_submission.csv', index_col= 0, header=0)
print(datasets.info())
print(datasets.describe())
print(datasets.isnull().sum())

############## 중복값 처리 #######################
print('중복값 제거 전', datasets.shape)
datasets = datasets.drop_duplicates()
print('중복값 제거 후', datasets.shape)

############## 이상치 확인 및 처리 ##################
outliers_loc = outliers(datasets['Garage Yr Blt'])
print('이상치의 위치 :', outliers_loc)
print('이상치 :',datasets.loc[[255], 'Garage Yr Blt'])   # 행과 열  // 2207(이상치)
datasets.drop(datasets[datasets['Garage Yr Blt']==2207].index, inplace = True)
print(datasets['Exter Qual'].value_counts())

'''
1사분위 :  1961.0
q2 : 1978.0
3사분위 : 2002.0
iqr : 41.0
이상치의 위치 : (array([254], dtype=int64),)
'''


print(datasets['Exter Qual'].value_counts())
'''
TA    808
Gd    485
Ex     49
Fa      8
'''
print(datasets['Kitchen Qual'].value_counts())
'''
TA    660
Gd    560
Ex    107
Fa     23
'''
print(datasets['Bsmt Qual'].value_counts())
'''
TA    605
Gd    582
Ex    134
Fa     28
Po      1
'''
print(test_sets['Exter Qual'].value_counts())
print(test_sets['Kitchen Qual'].value_counts())
print(test_sets['Bsmt Qual'].value_counts())
'''
test 파일에도 po가 있기 때문에 처리하는 방식에 대해 고민해봐야 한다. 
'''
# 품질 관련 변수 → 숫자로 매핑
qual_cols = datasets.dtypes[datasets.dtypes == np.object].index
def label_encoder(df_, qual_cols):
  df = df_.copy()
  mapping={
      'Ex':5, 'Gd':4, 'TA':3, 'Fa':2, 'Po':2
  }
  for col in qual_cols :
    df[col] = df[col].map(mapping)
  return df

datasets = label_encoder(datasets, qual_cols)
test_sets = label_encoder(test_sets, qual_cols)
datasets.head()
print(datasets.shape)  # (1350, 14)
print(test_sets.shape) # (1350, 13)

############################### 분류형 컬럼을 one hot encoding ###########################
datasets = pd.get_dummies(datasets, columns=['Exter Qual', 'Kitchen Qual', 'Bsmt Qual'])
test_sets = pd.get_dummies(test_sets, columns=['Exter Qual', 'Kitchen Qual', 'Bsmt Qual'])
############################### 분류형 컬럼을 one hot encoding ###########################
# print(datasets.columns)
# print(test_sets.columns)
print(datasets.shape)  # (1350, 23)
print(test_sets.shape) # (1350, 22)

########## xy분리 
x = datasets.drop(['target'], axis=1)
y = datasets['target']

test_sets = test_sets.values   # numpy로 변환
x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle = True, train_size = 0.8, random_state = 66
)

print(x_train.shape, y_train.shape)     # (1080, 22) (1080,)

scaler = StandardScaler()
# scaler = MinMaxScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()
# scaler = QuantileTransformer()
# scaler = PowerTransformer(method = 'box-cox')
# scaler = PowerTransformer(method = 'yeo-johnson')  # 디폴트

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
test_sets = scaler.transform(test_sets)

'''
{'target': 0.08937360852584253, 
'params': {'colsample_bytree': 0.82311593825603, 
'learning_rate': 0.11520475945944177, 
'max_depth': 6.9197971182170015, 
'min_child_weight': 2.0886940750281138, 
'n_estimators': 8293.453105466275, 
'reg_lamda': 6.869905678724954, 
'subsample': 0.7027705719420436}}
'''

colsample_bytree = 0.8231
learning_rate = 0.1151
max_depth = 7
min_child_weight= 2.0886
n_estimators= 8293
reg_lamda= 6.869
subsample= 0.7027

####################### 여기부터 SelectFromModel #####################
model = XGBRegressor(n_jobs=-1,
                     colsample_bytree = colsample_bytree,
                     learning_rate = learning_rate,
                     max_depth = max_depth,
                     min_child_weight = min_child_weight,
                     n_estimators = n_estimators,
                     reg_lamda = reg_lamda,
                     subsample = subsample)
model.fit(x_train, y_train,
          early_stopping_rounds = 100,
          eval_set=[(x_test, y_test)],
          eval_metric = 'mae')

#################  SelectFromModel ####################

# tresholds = np.sort(model.feature_importances_)
tresholds = model.feature_importances_

print(tresholds)

##################################불 필요 컬럼 삭제#################################
x_train = np.delete(x_train, [11,13,14,16], axis = 1)
x_test = np.delete(x_test, [11,13,14,16], axis = 1)
test_sets = np.delete(test_sets, [11,13,14,16], axis = 1)

print(x_train.shape, x_test.shape, test_sets.shape)
# (1078, 18) (270, 18) (1350, 18)
##################################불 필요 컬럼 삭제#################################


for thresh in tresholds:
  selection = SelectFromModel(model, threshold=thresh, prefit=True)
  
  select_x_train = selection.transform(x_train)
  select_x_test = selection.transform(x_test)
  print(select_x_train.shape,select_x_test.shape)
  
  selection_model = XGBRegressor(n_jobs=-1,
                     colsample_bytree = colsample_bytree,
                     learning_rate = learning_rate,
                     max_depth = max_depth,
                     min_child_weight = min_child_weight,
                     n_estimators = n_estimators,
                    #  reg_lamda = reg_lamda,
                     subsample = subsample)
  
  selection_model.fit(select_x_train, y_train,
          early_stopping_rounds = 100,
          eval_set=[(select_x_test, y_test)],
          eval_metric = 'mae', 
          verbose = 0)
  
  y_pred = selection_model.predict(select_x_test)
  
  score = r2_score(y_test, y_pred)
  
  print('Thresh = %.3f, n=%d, R2: %.2f%%' %(thresh, select_x_train.shape[1], score*100))
  


'''
(1078, 1) (270, 1)
Thresh = 0.280, n=1, R2: 70.46%
(1078, 5) (270, 5)
Thresh = 0.030, n=5, R2: 84.71%
(1078, 2) (270, 2)
Thresh = 0.260, n=2, R2: 75.84%
(1078, 19) (270, 19)
Thresh = 0.005, n=19, R2: 89.65%
(1078, 12) (270, 12)
Thresh = 0.010, n=12, R2: 88.82%
(1078, 11) (270, 11)
Thresh = 0.011, n=11, R2: 89.51%
(1078, 8) (270, 8)
Thresh = 0.015, n=8, R2: 84.85%
(1078, 10) (270, 10)
Thresh = 0.012, n=10, R2: 87.31%
(1078, 15) (270, 15)
Thresh = 0.008, n=15, R2: 88.39%
(1078, 20) (270, 20)
Thresh = 0.005, n=20, R2: 89.62%
(1078, 18) (270, 18)
Thresh = 0.005, n=18, R2: 89.19%
(1078, 9) (270, 9)
Thresh = 0.014, n=9, R2: 84.37%
(1078, 7) (270, 7)
Thresh = 0.018, n=7, R2: 84.80%
(1078, 22) (270, 22)
Thresh = 0.003, n=22, R2: 89.09%
(1078, 21) (270, 21)
Thresh = 0.003, n=21, R2: 88.72%
(1078, 14) (270, 14)
Thresh = 0.009, n=14, R2: 88.99%
(1078, 17) (270, 17)
Thresh = 0.006, n=17, R2: 88.37%
(1078, 16) (270, 16)
Thresh = 0.007, n=16, R2: 87.65%
(1078, 13) (270, 13)
Thresh = 0.009, n=13, R2: 89.05%
(1078, 4) (270, 4)
Thresh = 0.031, n=4, R2: 74.02%
(1078, 6) (270, 6)
Thresh = 0.020, n=6, R2: 84.54%
(1078, 3) (270, 3)
Thresh = 0.238, n=3, R2: 75.23%
'''
'''
Index(['Overall Qual', 'Gr Liv Area', 'Garage Cars', 'Garage Area',
       'Total Bsmt SF', '1st Flr SF', 'Full Bath', 'Year Built',
       'Year Remod/Add', 'Garage Yr Blt', 'target', 'Exter Qual_2',
       'Exter Qual_3', 'Exter Qual_4', 'Exter Qual_5', 'Kitchen Qual_2',
       'Kitchen Qual_3', 'Kitchen Qual_4', 'Kitchen Qual_5', 'Bsmt Qual_2',
       'Bsmt Qual_3', 'Bsmt Qual_4', 'Bsmt Qual_5']
       
[0.28014547 0.02980622 0.26040328 0.00476726 0.01023544 0.01089026
 0.01496887 0.01205076 0.00790177 0.00450289 0.00494695 0.01422107
 0.0184145  0.00302098 0.00330032 0.00883562 0.0058591  0.00732412
 0.0094856  0.03089884 0.02005105 0.2379696 ]
'''
'''
11, 13, 14, 16
'''