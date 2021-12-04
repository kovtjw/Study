import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import patches


path = '../_data/dacon/wine/'  
train = pd.read_csv(path+'train.csv')
train.head()

print(train)