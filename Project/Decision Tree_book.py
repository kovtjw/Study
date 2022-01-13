import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

path = '../_data/project data/' 
input_file = pd.read_csv(path +"광주 .csv")

x = input_file.drop(['일자','가격'], axis = 1)
y = input_file['가격']
x = np.array(x)

class_0 = np.array(x[y==0])
class_1 = np.array(x[y==1])
1