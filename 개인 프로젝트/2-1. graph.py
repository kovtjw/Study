import pandas as pd
import numpy as np
from matplotlib import font_manager, rc
import matplotlib.pyplot as plt

path = '../_data/project data/' 
gwangju = pd.read_csv(path + "광주.csv")
gwangju['일자'] = pd.to_datetime(gwangju['일자'])

font_path = "C:/Windows/Fonts/malgunbd.ttf"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)

plt.figure(figsize=(10,5))
plt.title('광주 감자 가격',font=font)
plt.plot(gwangju['일자'],gwangju['가격'],'r-',color='blue')
plt.xlabel('일자', font=font)
plt.ylabel('가격 ', font=font)
plt.legend(['가격'],loc='upper right', ncol=2, fontsize=10)
# plt.grid()
plt.show()


