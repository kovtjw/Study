import pandas as pd
import numpy as np

path = './_data/titanic/'
train = pd.read_csv(path + 'train.csv')
test = pd.read_csv(path + 'test.csv')
train.head()

print('train data shape: ', train.shape)
print('test data shape: ', test.shape)
print('----------[train infomation]----------')
print(train.info())
print('----------[test infomation]----------')
print(test.info())

import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set() # setting seaborn default for plots