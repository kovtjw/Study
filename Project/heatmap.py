from seaborn.matrix import heatmap
import xgboost
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score
import pandas as pd
import pandas as np
from xgboost.sklearn import XGBRegressor
import warnings
warnings.filterwarnings('ignore')
from matplotlib import font_manager, rc 
from xgboost import plot_importance
import seaborn as sns
import matplotlib.pyplot as plt
 

path = '../_data/project data/' 
gwangju = pd.read_csv(path +"gwangju .csv")
dataset = gwangju.drop(['일자'],axis = 1)
dataset = np.array(dataset)
#print(dft.info())

# dft=dft.drop(['지점','DATE','MAX TEMP(℃)','MIN TEMP(℃)','INSPECTION'], axis=1)

# print(dataset.info())

colormap=plt.cm.PuBu
plt.figure(figsize=(20,20))
plt.title("TEMP Correlation of Fteatures", y=1.00, size=15)
sns.heatmap(dataset.astype(float).corr(), linewidths=0.08, vmax=1.0, square=True, cmap=colormap, linecolor="white", annot=True, annot_kws={"size":6})

plt.show()