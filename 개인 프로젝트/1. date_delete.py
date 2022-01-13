import pandas as pd
import pandas as np

path = '../_data/project data/' 
kion = pd.read_csv(path +"kion.csv")

kion = kion.drop(pd.date_range('2016.1.1', '2021.12.31', freq='W-SAT'),axis=0)  # 기간 내 토요일 제외
kion = kion.drop(pd.date_range('2016.1.1', '2021.12.31', freq='W-SUN'),axis=0)  # 기간 내 일요일 제외

import holidays

kr_holidays = holidays.KR(years=[2016,2021])# 기간 내 농산물 가격 데이터가 없는 공휴일,임시공휴일을 추출

date_list=[]

for date, occasion in kr_holidays.items():
    date = (f'{date}')
    date_list.append(date)
    date_list.sort()
  
print(date_list)
'''
['2020-01-01', '2020-01-24', '2020-01-25', '2020-01-26', '2020-01-27', '2020-03-01', '2020-04-30', '2020-05-01', '2020-05-05', 
'2020-06-06', '2020-08-15', '2020-08-17', '2020-09-30', '2020-10-01', '2020-10-02', '2020-10-03', '2020-10-09', '2020-12-25', 
'2021-01-01', '2021-02-11', '2021-02-12', '2021-02-13', '2021-03-01', '2021-05-01', '2021-05-05', '2021-05-19', '2021-06-06', 
'2021-08-15', '2021-08-16', '2021-09-20', '2021-09-21', '2021-09-22', '2021-10-03', '2021-10-04', '2021-10-09', '2021-10-11', '2021-12-25']
'''
date_list.remove('2020-01-01')
date_list.remove('2020-01-25')
date_list.remove('2020-01-26')
date_list.remove('2020-03-01')
date_list.remove('2020-06-06')
date_list.remove('2020-08-15')
date_list.remove('2020-10-03')
date_list.remove('2021-02-13')
date_list.remove('2021-05-01')
date_list.remove('2021-06-06')
date_list.remove('2021-08-15')
date_list.remove('2021-10-03')
date_list.remove('2021-10-09')
date_list.remove('2021-12-25')

print(date_list)
'''
['2020-01-24', '2020-01-27', '2020-04-30', '2020-05-01', '2020-05-05', '2020-08-17', '2020-09-30', '2020-10-01', '2020-10-02',
'2021-01-01', '2021-02-11', '2021-02-12', '2021-03-01', '2021-05-05', '2021-05-19', '2021-08-16', '2021-09-20', '2021-09-21',
'2021-09-22', '2021-10-04', '2021-10-11']
'''
kion.drop(date_list[0:], axis=0, inplace=True)

kion.drop("2020-04-15", axis=0, inplace=True)  
kion.drop("2020-12-31", axis=0, inplace=True)  
kion.drop("2021-12-31", axis=0, inplace=True)  
# x1.drop("2020-10-09", axis=0, inplace=True)
