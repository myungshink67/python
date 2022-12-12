# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 08:37:49 2022

@author: KITCOOP
20221212.py
"""

'''
  범주형 데이터 : 값의 범위를 가진 데이터. 
                describe() 함수에서 조회시 제외.
  날짜 데이터 : pandas.date_range() : 날짜값을 범위 지정해서 조회
               df["Date"] : datetime 형
               df["Date"].dt.year : 년도
               df["Date"].dt.month : 월
               df["Date"].dt.day : 일
               
  형변환 : astype("자료형")   : str,int,float,category....    
  
  str : DataFrame의 요소들을 문자열처럼 사용. 문자열 함수 사용가능
              df["aaa"].str.startsWidth("")...     
              
  === 그룹화 : DataFrame을 컬럼의 값으로 데이터 분리
  groupby(컬럼명) : DataFrame 객체를 컬럼명의 값으로 분리.
  agg(함수)      : 지정한 함수를 적용할 수 있도록 하는 함수. 
                  사용자정의함수 사용가능
  filter(조건함수) : 조건함수의 결과가 참인 경우인 데이터 추출
  
  === 병합 : 두개의 DataFrame 연결
  concat : 물리적을 연결. 병합의미는 아님. 
  merge  : 연결컬럼의 값을 기준으로 같은 값은 가진 레코드들을 연결
            merge(df1,df2,on="연결컬럼",[how="outer/left/right"])
           두개의 데이터의 연결 컬럼명이 다른 경우 
            merge(df1,df2,left_on="왼쪽데이터연결컬럼",
                          right_on="오른쪽데이터연결컬럼"
                  [how="outer/left/right"])
'''
############
#  BeautifulSoup : html, xml 태그 분석 모듈
from bs4 import BeautifulSoup #html,xml 파싱 모듈
import urllib.request as req
url="https://www.weather.go.kr/weather/forecast/mid-term-rss3.jsp"
res=req.urlopen(url)
soup = BeautifulSoup(res,"html.parser")
title=soup.find("title").string #title 태그 선택
wf=soup.find("wf").string  #wf 태그 선택
title
wf

#wf 데이터를 <br /> 문자열로 분리하여 화면에 출력하기

