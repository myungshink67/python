# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 13:51:25 2022

@author: KITCOOP
test1212_a.py
"""

'''
1. http://www.kma.go.kr/weather/forecast/mid-term-rss3.jsp 의 내용을 
   인터넷을 통해 데이터를 수신하고 다음 결과형태로 출력하시오.
   결과는 현재 날씨에 따라 달라 질수 있습니다.
[결과]
+ 구름많고 눈
 |-  서울
 |-  인천
 |-  수원
 |-  파주
 |-  이천
 |-  평택
+ 구름많음
 |-  춘천
 |-  원주
 |-  청주
 |-  충주
 |-  영동
 |-  광주
 |-  목포
 |-  여수
 |-  순천
 |-  광양
 |-  나주
 |-  대구
 |-  안동
 |-  포항
 |-  경주
 |-  울진
 |-  울릉도
 |-  제주
 |-  서귀포
+ 맑음
 |-  강릉
 |-  부산
 |-  울산
 |-  창원
 |-  진주
 |-  거창
 |-  통영
+ 구름많고 비/눈
 |-  대전
 |-  세종
 |-  홍성
+ 흐리고 비/눈
 |-  전주
 |-  군산
 |-  정읍
 |-  남원
 |-  고창
 |-  무주
'''   

from bs4 import BeautifulSoup
import urllib.request as req
url="https://www.kma.go.kr/weather/forecast/mid-term-rss3.jsp"
res = req.urlopen(url)
info = {}  #딕셔너리. {날씨 : 도시목록}
#xml 형태의 결과를 분석하기
soup = BeautifulSoup (res, "html.parser")
#soup.find_all("location") : location 태그들
for location in soup.find_all("location") :
    name = location.find("city").string 
    weather = location.find("wf").string
#weather in info : 
#    info 딕셔너리객체의 키값에 weather 값 존재?
    if not (weather in info) : 
        info[weather] = []  #리스트객체 생성.
    info[weather].append(name) #도시명 추가
#화면출력
for weather in info.keys() :
    print("+",weather)
    for name in info[weather] : 
        print(" |- ",name)
 
'''
2. 네이버 로그인 후 상품 목록 조회하기 
   네이버 로그인.
   쇼핑 메뉴 선택
   쇼핑MY 메뉴 선택
   주문확인/배송조회를 선택
'''
from selenium import webdriver
from selenium.webdriver.common.by import By
import time
driver = webdriver.Chrome("D:/setup/chromedriver")
driver.get("https://nid.naver.com/nidlogin.login?mode=form&url=https%3A%2F%2Fwww.naver.com")
id = input("네이버 아이디를 입력하세요:")
driver.execute_script("document.getElementsByName('id')[0].value='"+id+"'")
pw = input("네이버 비밀번호를 입력하세요 : ")
time.sleep(1) 
driver.execute_script\
("document.getElementsByName('pw')[0].value='"+pw+"'")
time.sleep(1)
driver.find_element('xpath','//*[@id="log.login"]').click()


# 쇼핑 영역 클릭하기
#driver.find_element('xpath','//*[@id="NM_FAVORITE"]/div[1]/ul[1]/li[5]/a').click()
driver.find_element('css selector','.nav.shop').click()
time.sleep(1)

#driver.find_element('xpath','//div[@class="_myButton_my_button_1GyxC"]').click()
driver.find_element('css selector','div._myButton_my_button_1GyxC').click()
time.sleep(1)

#주문확인/배송조회 클릭하기
#driver.find_element('xpath','//*[@class="_myButton_layer_my_TuzvW"]/ul/li[6]/a').click()
#driver.find_element(By.LINK_TEXT,'주문확인/배송조회').click()
driver.find_element(By.PARTIAL_LINK_TEXT,'주문확인').click()

time.sleep(1)

#주문정보 보기
products = driver.find_elements('css selector',".goods_pay_section")
for product in products:
    print("-", product.text)
time.sleep(2)   
driver.quit()