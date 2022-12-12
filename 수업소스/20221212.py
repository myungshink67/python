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
  agg(함수)      : 지정한 함수를 적용할 수 있도록 하는 함수. 사용자정의함수 사용가능
  filter(조건함수) : 조건함수의 결과가 참인 경우인 데이터 추출
  
  === 병합 : 두개의 DataFrame 연결
  concat : 물리적을 연결
  merge  : 연결컬럼의 값을 기준으로 같은 값은 가진 레코드들을 연결
            merge(df1,df2,on="연결컬럼",how="outer/left/right")
'''
