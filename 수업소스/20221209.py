# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 16:50:57 2022

@author: KITCOOP
20221209.py
"""

'''
  데이터 전처리 : 원본데이터를 원하는 형태로 변경 과정. 
    1. 결측값 처리 : 값이 없는 경우.
        - isnull() : 결측값인 경우 True, 일반값인 경우 False
        - notnull() : 결측값인 경우 False, 일반값인 경우 True
        - dropna() : 결측값 제거 함수
              dropna(axis=1,thresh=500) : 결측값이 500개 이상인 컬럼 제거
              dropna(subset=[컬럼명],how=any/all, axis=0) :결측값을 가지고 있는 행 제거
                                              any:한개라도 결측값.
                                              all:모두 결측값
        - fillna() :결측값 치환
           fillna(치환값,inplace=True)
           fillna(방법,inplace=True) : method="ffill"/"bfill" :앞의값/뒤의값
           
    2. 중복데이터 처리 
       - duplicated() : 중복데이터 찾기. 첫번째 데이터는 False, 
                        같은 데이터인 경우 두번째 True       
       - drop_duplicates() : 중복데이터를 제거. 중복된 데이터 중 한개는 남김.

    3. 새로운 컬럼 생성
       - df[컬럼명] : 컬럼명이 없는 경우 생성, 있으면 수정.
       - round(자리수) : 반올림.    

    4. 오류데이터 존재.
       - 결측값 치환 : 결측값(np.nan) 
                 replace(오류문자열, np.nan, inplace=True)
'''     
