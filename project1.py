import numpy as np
import pandas as pd 
import datetime
import time
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense,LSTM, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

path = '../_data/project data/' 
gwangju = pd.read_csv(path +"광주 .csv")
dataset = gwangju.drop(['일자'],axis = 1)
gwangju = np.array(gwangju)

def split_xy(dataset, time_steps, y_column):
    x, y = list(), list()
    for i in range(len(dataset)):
        x_end_number = i + time_steps
        y_end_number = x_end_number + y_column - 1
        
        if y_end_number > len(dataset):
            break
        tmp_x = dataset[i:x_end_number, : -1]
        tmp_y = dataset[x_end_number-1 : y_end_number, -1]
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)

x, y = split_xy(dataset, 3, 4)

print(x.shape)
print(y.shape)





