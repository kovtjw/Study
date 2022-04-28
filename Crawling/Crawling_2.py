# selenium 관련 import
from selenium.webdriver.common.keys import Keys
from selenium import webdriver
import urllib.request,time,warnings,os     # url주소,경고창,os폴더생성

warnings.filterwarnings(action='ignore')    # 경고 무시
###################################사전설정##################################################
keyword = input("검색할 단어 : ") # 구글에 검색할 keyword
saveword = input('저장할 단어 : ') # 저장할 단어

# 원하는 장수만큼 사진 저장
n = int(input('저장 할 사진 장수를 입력해 주세요 : '))
ne = round(n/33)    #number error 에러로 몇 장 저장 못할 수 있으니 미리 몇 장 더해준다.

# 폴더 생성하기
path = os.path.dirname(os.path.realpath(__file__))
os.makedirs(f'{path}/data/{saveword}',exist_ok=True)                  # 현재경로를 따와서 data 폴더 만들고 그 안에 마동석 폴더생성


##################################크롤링 시작##################################################

# 크롬창에서 F12누르면 html 작업창이 나옴.
# 크롬드라이버 옵션 설정, 2번째줄은 보안관련 해제해주는 옵션
options = webdriver.ChromeOptions()
options.add_experimental_option('excludeSwitches', ['enable-logging'])
driver = webdriver.Chrome('C://chromedriver.exe', options=options)

# 크롬실행 후 검색창에 마동석 입력
# keyword = input('검색할 키워드를 입력하세요(exit -> 종료) : ')    # 자동화 할 경우


driver.get("https://www.google.co.kr/imghp?hl=ko&ogbl")           # 실행할 창 주소 넣어주세요.
elem = driver.find_element_by_name("q")                           # name으로 검색창 element 이름 가져와서 elem에 저장                        
elem.send_keys(keyword)                                          # 검색창에 마동석이라고 key를 보냅니다.
elem.send_keys(Keys.RETURN)                                       # 엔터누르는 작업


driver.find_element_by_xpath('/html/body/div[3]/c-wiz/div[1]/div/div[1]/div[2]/div[2]/div').click()      # 도구-모든크기-큼
driver.find_element_by_xpath('/html/body/div[3]/c-wiz/div[2]/div[2]/c-wiz[1]/div/div/div[1]/div/div[1]/div/div[1]').click()
driver.find_element_by_xpath('/html/body/div[3]/c-wiz/div[2]/div[2]/c-wiz[1]/div/div/div[3]/div/a[2]/div').click()

images = driver.find_elements_by_css_selector(".rg_i.Q4LuWd")     # 검색 후 화면에서 나오는 사진들 목록을 가져옴

if len(images) < n+ne:                                                  # 만약 내가 다운받고 싶은 사진이 로드 된 이미지 수보다 적다면.                                                    
        while True:                                                         # 무한대로 실행하겠다.                             
            last_height = driver.execute_script("return document.body.scrollHeight")                   
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")                    
            time.sleep(1)                                                                                    
            new_height = driver.execute_script("return document.body.scrollHeight")                    
            images = driver.find_elements_by_css_selector(".rg_i.Q4LuWd")               # 스크롤을 내리고 사진 장수를 다시 구함.                     
            print(f'{keyword}사진 현재{len(images)}장 로드, {n+ne}장까지 로드하기 위해 스크롤 중................') 
            
            if len(images) >= n+ne:                          # 로드된 사진이 내가 원하는 사진보다 많다면.                                                            
                print(f'.........................................현재{len(images)}장까지 로드하여 이제 저장을 시작합니다')
                break   # 끊겠다.                                                                                    
            else:       # 아니라면 스크롤을 또 내림                                                                               
                if new_height == last_height:       # scroll 높이가 갱신이 안된경우 > 결과 더보기 버튼 클릭해야 내려간다.                                                           
                    try:                                                                                        
                        driver.find_element_by_css_selector('.mye4qd').click()    # 결과 더보기 버튼을 클릭해라.                      
                    except:                                                                                     
                        break                                                        # 끊겠다.                            
                last_height = new_height             

count = 1 # 다운받는 사진 개수 count하기 위한 변수

totalstart = time.time()

for image in images:   # 새로 load된 images 목록을 for문 함
    try:
        image.click()                    # 1장의 Image click
        start = time.time()              # 1장 다운받는데 몇 초 걸리나 계산하기 위해
        time.sleep(1)                    # 고화질 시간을 로드ㅜ하기 위해 1초 여유시간 줌 
        imgUrl = driver.find_element_by_xpath('/html/body/div[3]/c-wiz/div[3]/div[2]/div[3]/div/div/div[3]/div[2]/c-wiz/div/div[1]/div[1]/div[3]/div/a/img').get_attribute('src')
        print(f'{count} / {n}, {keyword} 사진 다운로드 중 .....Download time :'+str(time.time()-start)[:5] + '초')
        urllib.request.urlretrieve(imgUrl, f"{path}/data/{saveword}/{count:0{len(str(n+ne))}d}{saveword}.jpg")
        
        if count == n:
            break
        count += 1
    except:
        print(f'{count}번째 사진 no 저장')
        
totalend = str(time.time()-totalstart)[:5]
    
if count < n:  # 1000장 다운받고 싶은데 500장 밖에 load가 안되는 경우
    print(f'더 이상의 사진이 없기 때문에 {n}장까지 다운로드하지 못하고 {count-1}장까지만 저장했습니다.')
    



print(f'----------다운로드 시간 {totalend}----------')
driver.close()
print(f'------------{saveword} 폴더 생성이 완료되었습니다. ')



# for i, image in enumerate(images):   # 48장의 image를 1장씩 가져오겠다
#     try:
#         image.click()                    # 1장의 Image click
#         time.sleep(1)
#         imgUrl = driver.find_element_by_xpath('/html/body/div[3]/c-wiz/div[3]/div[2]/div[3]/div/div/div[3]/div[2]/c-wiz/div/div[1]/div[1]/div[3]/div/a/img').get_attribute('src')
#         urllib.request.urlretrieve(imgUrl, f"{path}/data/{saveword}/{saveword}_{i:02d}.jpg")
#     except:
#         print('{i}번째 사진 no 저장')
    
    
    