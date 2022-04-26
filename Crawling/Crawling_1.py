from selenium.webdriver.common.keys import Keys
from selenium import webdriver
import urllib.request
import warnings
warnings.filterwarnings(action='ignore')
import time

options = webdriver.ChromeOptions()
options.add_experimental_option('excludeSwitches', ['enable-logging'])
driver = webdriver.Chrome('C://chromedriver.exe', options=options)
driver.get("https://www.google.co.kr/imghp?hl=ko&ogbl")

elem = driver.find_element_by_name("q")
elem.send_keys(f'마동석')
elem.send_keys(Keys.RETURN)





# time.sleep(5)
