# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 16:01:58 2022

@author: KITCOOP
20221202.py
"""

'''
pandas 모듈
 - 표형태(행:index,열:columns)의 데이터를 처리하기 위한 모듈
 - Series : 1차원형태의 데이터처리. DataFrame의 한개의 컬럼값들의 자료형
 - DataFrame : 표형태의 데이터처리. Series데이터의 모임.
     - 기술통계함수 : sum,mean,median,max,min,std,var,describe
     - 행의 값 : index
     - 열의 값 : columns
     - rename : index,columns의 값을 변경 함수 inplace=True : 객체자체변경
     - drop   : index,columns의 값을 제거 함수 inplace=True : 객체자체변경
     - 얕은복사 : df2 = df, df,df2객체는 물리적으로 같은 객체 
     - 깊은복사 : df3=df[:], df4=df.copy()
     
     - 한개의 컬럼조회 : df["컬럼명"], df.컬럼명 => Series 객체
     - 여러개의 컬럼조회 : df[["컬럼1","컬럼2"]] => DataFrame 객체
                          df["row1":"rown"] =>  DataFrame 객체. 범위지정
     - 행을 조회 : loc["인덱스명"], iloc[인덱스순서]                     
     - 컬럼 => 인덱스 : set_index
     - 인덱스 => 컬럼 : reset_index
     - csv 파일 읽기 : read_csv                     
     - csv 파일 쓰기 : to_csv                     
     - excel 파일 읽기 : read_excel
     - excel 파일 쓰기 : ExcelWriter > to_excel > save
'''
