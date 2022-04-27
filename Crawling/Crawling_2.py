# selenium 관련 import
from selenium.webdriver.common.keys import Keys
from selenium import webdriver
import urllib.request,time,warnings,os     # url주소,경고창,os폴더생성

warnings.filterwarnings(action='ignore')    # 경고 무시

# 크롬창에서 F12누르면 html 작업창이 나옴.
# 크롬드라이버 옵션 설정, 2번째줄은 보안관련 해제해주는 옵션
options = webdriver.ChromeOptions()
options.add_experimental_option('excludeSwitches', ['enable-logging'])
driver = webdriver.Chrome('C://chromedriver.exe', options=options)

# 크롬실행 후 검색창에 마동석 입력
# keyword = input('검색할 키워드를 입력하세요(exit -> 종료) : ')    # 자동화 할 경우

driver.get("https://www.google.co.kr/imghp?hl=ko&ogbl")           # 실행할 창 주소 넣어주세요.
elem = driver.find_element_by_name("q")                           # name으로 검색창 element 이름 가져와서 elem에 저장                        
elem.send_keys('마동석')                                          # 검색창에 마동석이라고 key를 보냅니다.
elem.send_keys(Keys.RETURN)                                       # 엔터누르는 작업
images = driver.find_elements_by_css_selector(".rg_i.Q4LuWd")     # 검색 후 화면에서 나오는 사진들 목록을 가져옴


# 폴더 생성하기
path = os.path.dirname(os.path.realpath(__file__))

os.makedirs(f'{path}/data/마동석',exist_ok=True)                  # 현재경로를 따와서 data 폴더 만들고 그 안에 마동석 폴더생성

# 일단 1장의 사진만 다운하기
images[0].click()       # 클릭해서 크게 사진 크게 키우기
# 키운사진에 대한 src주소값을 가져옴
imgUrl = driver.find_element_by_xpath('/html/body/div[3]/c-wiz/div[3]/div[2]/div[3]/div/div/div[3]/div[2]/c-wiz/div/div[1]/div[1]/div[3]/div/a/img').get_attribute('src') 
urllib.request.urlretrieve(imgUrl, f"{path}/data/마동석/haha.jpg")    # url주소와 저장경로 입력

time.sleep(3)
