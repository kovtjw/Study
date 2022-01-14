from sklearn.metrics import r2_score, mean_squared_error #mse
from sklearn.model_selection import train_test_split
import pandas as pd 

#1. 데이터 
path = '../_data/kaggle/bike/'  
train = pd.read_csv(path+'train.csv')  
test_file = pd.read_csv(path+'test.csv')
