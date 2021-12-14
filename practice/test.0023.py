import numpy as np
import pandas as pd
import time
from sklearn.preprocessing import RobustScaler, MaxAbsScaler, LabelEncoder, OneHotEncoder
import matplotlib.pyplot as plt

#1 데이터
path = "D:\\Study\\_data\\dacon\\wine\\"
train = pd.read_csv(path +"train.csv")
test_file = pd.read_csv(path + "test.csv") 
submission = pd.read_csv(path+"sample_Submission.csv") #제출할 값

# print(train)

y = train['quality']
x = train.drop(['id','quality'], axis =1)
x = x.drop(['citric acid','pH','sulphates','total sulfur dioxide'],axis =1)
test_file =test_file.drop(['id','citric acid','pH','sulphates','total sulfur dioxide'],axis =1)
le = LabelEncoder()
le.fit(train['type'])
x['type'] = le.transform(train['type'])

le.fit(test_file['type'])
test_file['type'] = le.transform(test_file['type'])

# y = np.array(y).reshape(-1,1)
# enc= OneHotEncoder()   #[0. 0. 1. 0. 0.]
# enc.fit(y)
# y = enc.transform(y).toarray()

y = y.to_numpy()
x = x.to_numpy()

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, 
         train_size = 0.8, shuffle = True, random_state = 66) #455.2 /114
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_validate

scaler = RobustScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
from sklearn.ensemble import RandomForestClassifier
model2 = RandomForestClassifier(random_state=66)
model2.fit(x_train, y_train)
y_pred2 = model2.predict(test_file)
print(y_pred2)

submission['quality'] = y_pred2
submission.to_csv("dacon_wine12.csv", index=False)