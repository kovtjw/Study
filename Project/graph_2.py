import pandas as pd
import numpy as np
from matplotlib import font_manager, rc
import matplotlib.pyplot as plt

path = '../_data/project data/' 
inchoen = pd.read_csv(path + "data_광주.csv")
# gwangju['일자'] = pd.to_datetime(gwangju['일자'])
inchoen['일자']= inchoen['일자'].astype('str')
inchoen['일자'] = pd.to_datetime(inchoen['일자'])

# print(inchoen.info())


font_path = "C:/Windows/Fonts/malgunbd.ttf"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)

plt.figure(figsize=(20,15))
plt.title('일자별 감자 가격', font=font)
plt.plot(inchoen['일자'],inchoen['가격'],marker='o', color='blue')
plt.xlabel('일자', font=font)
plt.ylabel('가격 ', font=font)
plt.legend(['가격'],loc='upper right', ncol=2, fontsize=20)
# plt.grid()
plt.show()