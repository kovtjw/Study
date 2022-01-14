from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import numpy as np
import pandas as pd

def create_model():
   model = Sequential()
   model.add(Dense(4, input_dim=8, activation='relu'))
   model.add(Dense(4, activation='relu'))
   model.add(Dense(1))
   
   model.compile(loss='mae', optimizer='adam')
   return model
 

seed = 7
np.random.seed(seed)
 
path = '../_data/project data/' 
gwangju = pd.read_csv(path +"data_광주.csv")
x = gwangju.drop(['일자','가격'], axis = 1)
y = gwangju['가격']
 
# create model
model = KerasClassifier(build_fn=create_model, epochs=150, batch_size=10, verbose=1)
 
kfold = KFold(n_splits=2, shuffle=True, random_state=seed)
results = cross_val_score(model, x, y, cv=kfold)


print(results)