from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
datasets = load_boston()

#1. 데이터 

x = datasets.data
y = datasets.target
import pandas as pd
xx = pd.DataFrame(x, columns=datasets.feature_names)
# print(type(xx))
# print(xx.corr())  # corr = 상관관계, 1이면 상관관계가 높음
xx['price'] = y
# print(xx)
import matplotlib.pyplot as plt
import seaborn as sns 
plt.figure(figsize=(10,10))
sns.heatmap(data=xx.corr(), squre=True, annot=True, cbar=True)
plt.show()