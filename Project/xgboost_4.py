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
from matplotlib import font_manager, rc
font_path = "C:/Windows/Fonts/malgunbd.ttf"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)
import seaborn as sns
from sklearn.model_selection import cross_val_score

path = '../_data/project data/' 
gwangju = pd.read_csv(path +"merge_gwangju.csv")
x = gwangju.drop(['일자','가격'], axis = 1)
y = gwangju['가격']

df4_x_train, df4_x_test, df4_y_train, df4_y_test=train_test_split(x, y, test_size=0.2)
xgb_model=xgboost.XGBRegressor(n_estimators=400, learning_rate=0.08, gamma=0, subsampel=0.5, colsample_bytree=1, max_depth=7)

xgb_model.fit(df4_x_train, df4_y_train)

xgboost.plot_importance(xgb_model)
fig, ax = plt.subplots(figsize=(10,12))
plot_importance(xgb_model, ax=ax)
plt.show()


y_pred = xgb_model.predict(df4_x_test)
print(y_pred)

# r_sq = xgb_model.score(df4_x_train, df4_y_train)
# loss = xgb_model.evaluate(df4_x_test)
# print(r_sq)
# print(explained_variance_score(pred, df4_y_test))
