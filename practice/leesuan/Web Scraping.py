
'''
# 웹 스크래핑
HTTP를 통햬 웹 사이트의 내용을 긁어, 원하는 형태로 가공하는 것
웹 사이트의 데이터를 수집하는 모든 작업을 의미

# 크롤링
웹 크롤러에서 유래, 조직적, 자동화된 방법으로 월드와이드웹을 탐색하는 프로그램
찾아낸 데이터를 저장한 후 쉽게 찾을 수 있게 인덱싱 수행

# 파싱
어떤 페이지(문서, html 등)에서 내가 원하는 데이터를 특정 패턴이나 순서로 추출하여 정보를 가공하는 것
일련의 문자열을 의미있는 토큰으로 분해하고 이들로 이루어진 파스트리를 만드는 과정
입력 토근에 내제된 자료 구조를 빌드하고 문법을 검사하는 역할을 함
'''

# 네이버 검색 API예제는 블로그를 비롯 전문자료까지 호출방법이 동일하므로 blog검색만 대표로 예제를 올렸습니다.
# 네이버 검색 Open API 예제 - 블로그 검색
import os
import sys
import urllib.request
client_id = "gTqNaqyjog14g27TrawW"
client_secret = "VrNLxqvvkm"
encText = urllib.parse.quote("손흥민")
url = "https://openapi.naver.com/v1/search/blog?query=" + encText # json 결과
# url = "https://openapi.naver.com/v1/search/blog.xml?query=" + encText # xml 결과
request = urllib.request.Request(url)
request.add_header("X-Naver-Client-Id",client_id)
request.add_header("X-Naver-Client-Secret",client_secret)
response = urllib.request.urlopen(request)
rescode = response.getcode()
if(rescode==200):
    response_body = response.read()
    print(response_body.decode('utf-8'))
else:
    print("Error Code:" + rescode)
    
    
import re
import json
import math
import request
import urllib.request
import urllib.error
import urllib.parse
from bs4 import Beautifulsoup