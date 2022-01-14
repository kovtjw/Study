import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense,LSTM, Input, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import mean_squared_error as MSE
import matplotlib.pyplot as plt
path = '../_data/project data/' 
gwangju = pd.read_csv(path +"data_광주.csv")
print(gwangju.info())


dataset = gwangju.drop(['일자'],axis = 1)
dataset = np.array(dataset)

