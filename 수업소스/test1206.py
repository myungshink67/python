# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 15:51:11 2022

@author: KITCOOP
test1206.py
"""

#1.seaborn 모듈의 iris 데이터 셋을 이용하여  품종별 산점도를 출력하기
# 20221206-1.png 파일 참조

#2. iris 데이터 셋을 이용하여  각 컬럼의 값을  박스그래프로 작성하기
# 20221206-2.png 파일 참조

#3. tips 데이터 셋의 total_bill 별 tip  컬럼의 회귀선을 출력하기
# 20221206-3.png 파일 참조

import seaborn as sns
tips = sns.load_dataset("tips")
tips.info()
#4. tips 데이터에서 점심,저녁별 tip 평균 금액을 막대그래프로 작성하기
# 20221206-4.png 파일 참조

#5. tips 데이터에서 점심,저녁별 건수를 막대그래프로 작성하기
# 20221206-5.png 파일 참조

'''
6. 서울시 범죄율 데이터를 이용하여 살인 정보를 지도에 표시하기
  지도 : 20221206-1.html 참조
  지도표시 데이터 : skorea_municipalities_geo_simple.json
  서울시 범죄율 데이터 : crime_in_Seoul_final.csv
'''