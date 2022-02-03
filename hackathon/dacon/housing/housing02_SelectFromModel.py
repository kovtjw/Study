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
learning_rate = 0.115
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

tresholds = np.sort(model.feature_importances_)
# tresholds = model.feature_importances_

print(tresholds)
print(datasets.columns)

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
                     reg_lamda = reg_lamda,
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
hresh = 0.004, n=22, R2: 89.45%
(1078, 21) (270, 21)
'''