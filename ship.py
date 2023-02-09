# 네이버에서 어선 사진 크롤링
# 2019. 12. 10. (금)

#-*- coding: utf-8 -*-
#%%
# 셀레니움 라이브러리
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from bs4 import BeautifulSoup
import time
import urllib.request
import os

from selenium import webdriver
from selenium.webdriver.common.by import By
from urllib import request    # 이미지 다운로드에 사용
from bs4 import BeautifulSoup as bs
from tqdm import tqdm
import time
import os


import pandas as pd
import numpy as np

# ----------------------------------------------------------------------------------------
#%%
#크롬 드라이버 경로
driver = webdriver.Chrome("chromedriver_win32/chromedriver.exe")



url="https://search.naver.com/search.naver?sm=tab_hty.top&where=image&query=%EC%96%B4%EC%84%A0+-%EC%9A%94%EB%A6%AC+-%EA%B2%8C%EC%9E%84+-%EB%A7%88%EB%B9%84%EB%85%B8%EA%B8%B0+-%EC%84%A0%EC%9B%90%EB%AA%A8%EC%A7%91&oquery=%EC%96%B4%EC%84%A0+-%EC%9A%94%EB%A6%AC+-%EA%B2%8C%EC%9E%84+-%EB%A7%88%EB%B9%84%EB%85%B8%EA%B8%B0&tqi=h%2B7rAsprvmsssbUwkAossssssx4-445680"

# url을 이용한 페이지 이동
driver.get(url)

#스크로 이동 횟수 정의
count_down=int(input("스크롤 다운 횟수 입력(최대:6):"))

# 사용자가 요구한 스크롤 다운 횟수 확인(최대 6회)
if count_down > 15:
    count_down = 15


# windows.scrollTo(0, Height) : 스크롤을 아래쪽으로 이동
# Height 대신 document.body.scrollHeight를 입력하면 페이지 끝으로 이동

driver.get(url)

#스크로 이동 횟수 정의
count_down=int(input("스크롤 다운 횟수 입력(최대:6):"))

# 사용자가 요구한 스크롤 다운 횟수 확인(최대 6회)
if count_down > 6:
    count_down = 6


# windows.scrollTo(0, Height) : 스크롤을 아래쪽으로 이동
# Height 대신 document.body.scrollHeight를 입력하면 페이지 끝으로 이동

for i in tqdm(range(count_down)):
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight)")
    time.sleep(1)

"""#### 화면에서 이미지 추출하기
1. 드라이브의 소스코드 받아오기
2. 이미지가 있는 태그 찾기
3. 태그에서 이미지 소스("src") 받아오기
4. src 주소를 이용해 이미지 다운로드 하기
"""

# 드라이브 소스코드 가져오기
html = driver.page_source
soup = bs(html, "html.parser")

"""#### Image lazy loading
- 현재 화면에 필요한 이미지만 불러오는 기능
- 구글에서 검색해 보세요
"""

# 이미지 대표 태그 찾아 tag_img 변수에 저장(tag_img => 리스트형 데이터)
tag_img = soup.find_all("div", class_="thumb")

img=tag_img[4].find("img", class_="_image _listImage")

if img.get("data-lazy-src") == None:
    print(img["src"])
else:
    print(img.get("data-lazy-src"))

# 태그에서 이미지 소스("src") 받아오기
tag_src=[]
for tag in tqdm(tag_img):
    img=tag.find("img", class_="_image _listImage")
    if img.get("data-lazy-src") == None:
        tag_src.append(tag.find("img")['src'])
    else:
        tag_src.append(img.get("data-lazy-src"))
        

#tag_src=[tag.find("img")['src'] for tag in tag_img]
#print(len(tag_src))
driver.close()

# 이미지 src를 이용해 이미지 데이터 다운로드 후 저장하기
img_name = "./pydata/" + keyword + ".jpg"

from urllib import request

# 웹상의 이미지를 다운로드후 저장
# urllib.request.urlretrieve(대상 src_url, "저장경로와 파일명") 
request.urlretrieve(tag_src[4], img_name)

"""#### 검색어를 이용해 폴더 생성후 전체 이미지 저장
- os.makedirs("폴더 경로/생성 폴더명") => 폴더 생성 명령어
- 폴더가 없으면 생성후 저장
- 기존에 폴더가 존재하며 기존 폴더이름 뒤에 1, 2, .. 와 같이 번호를 붙여 생성 

"""

# 폴더 생성 실습

# 이미지를 저자할 폴더 경로
fdir = "./pydata/"

if os.path.exists(fdir):  # 폴더가 있다면 뒤쪽에 "/"만 연결
    fdir += "/"
else:
    os.makedirs(fdir)     # 폴더가 없다면 폴더 생성후 뒤쪽에 "/"만 연결
    fdir += "/"

# images 폴더에 검색 키워드를 이용해 폴더 생성후 저장
# 키워드와 동일한 폴더가 있는 경우 뒤에 1씩 번호를 증가시기면서 폴더명 확인(없으면 생성)

if not os.path.exists(fdir + keyword):
    os.makedirs(fdir + keyword)     # 폴더가 없다면 폴더 생성후 뒤쪽에 "/"만 연결
    fdir = fdir + keyword +  "/"
else:
    # 폴더가 있다면 새로운 폴더 생성(번호 증가)
    num = 0
    while True:
        num += 1 #번호 1씩 증가
        
        #증가된 번호와 기존 폴더명을 연경해서 존재여부 확인
        if not os.path.exists(fdir + keyword + str(num)):
            # 없으면 폴더 생성후 while 종료
            os.makedirs(fdir + keyword + str(num)) 
            fdir = fdir + keyword + str(num) + "/"
            break

"""이미지 저장
- 저장 이미지 이름 뒤에 번호 붙이기
"""

num = 1
for tag in tqdm(tag_src):
    img_name = fdir + keyword + str(num) + ".jpg"
    request.urlretrieve(tag, img_name)
    #time.sleep(1)
    num += 1