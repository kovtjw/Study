from pytimekr import pytimekr

chuseok= pytimekr.chuseok()
print(chuseok)

# 라이브러리 호출
from pytimekr import pytimekr

# 추석
pytimekr.chuseok()

# 설날
pytimekr.lunar_newyear()

# 한글날
pytimekr.hangul()

# 어린이날
pytimekr.children()

# 광복절
pytimekr.independence()

# 현충일
pytimekr.memorial()

# 석가탄신일
pytimekr.buddha()

# 삼일절
pytimekr.samiljeol()

# 제헌절
pytimekr.constitution()

# 연도별 추석 날짜 구하기
for i in range(2010, 2021):
    chuseok = pytimekr.chuseok(i)
    print(chuseok)