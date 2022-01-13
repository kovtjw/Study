import numpy as np
import pandas as pd 
import matplotlib
import pandas as pd
import numpy as np
import numpy as np
from numpy.core.fromnumeric import shape
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint 
import time
from tensorflow.keras.layers import Dropout, Dense, SimpleRNN,LSTM
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
import matplotlib as mpl
import csv
import matplotlib.font_manager as fm
import matplotlib_inline


path = '../_data/project data/' 
gwangju = pd.read_csv(path + "gwangju .csv")

# x = gwangju.drop(['기온','강수량','습도','풍량'], axis = 1)
# gwangju['일자']= gwangju['일자'].astype('str')
gwangju['일자'] = pd.to_datetime(gwangju['일자'])


from matplotlib import font_manager, rc
font_path = "C:/Windows/Fonts/malgunbd.ttf"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)

plt.figure(figsize=(100,100))
plt.title('일자별 감자 가격', font=font)
plt.plot(gwangju['일자'],gwangju['가격'],'r-',marker='s', color='blue')
plt.xlabel('일자', font=font)
plt.ylabel('가격 ', font=font)
plt.legend(['가격'],loc='upper right', ncol=2, fontsize=20)
# plt.grid()
plt.show()

# x = gwangju.drop(['일자','가격','풍량','기온'], axis = 1)
# y = gwangju['가격']

