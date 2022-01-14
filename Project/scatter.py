import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
font_path = "C:/Windows/Fonts/malgunbd.ttf"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)

path = '../_data/project data/' 
df = pd.read_csv(path + "data_대구 .csv")
df['일자'] = pd.to_datetime(df['일자'])
df['일자']= df['일자'].astype('str')
plt.scatter(x=df['일자'], y = df['가격'])
plt.xlabel('일자', fontsize = 12)
plt.ylabel('가격', fontsize = 12)
plt.show()