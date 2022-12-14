# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 08:52:49 2022

@author: KITCOOP
20221214.py
"""

###################
#  전세계 음주 데이터 분석하기 : drinks.csv
import pandas as pd
drinks = pd.read_csv("data/drinks.csv")
drinks.info()
#continent 컬럼의  데이터가 결측값인 'OT'으로 치환
drinks["continent"] = drinks["continent"].fillna('OT')
drinks.info()

#대한민국은 얼마나 술을 독하게 마시는 나라인가?
#total_servings : 전체 주류 소비량 컬럼 추가
drinks["total_servings"] =\
    drinks["beer_servings"] + \
    drinks["spirit_servings"] +\
    drinks["wine_servings"]
 
#alcohol_rate : 알콜비율 (알콜섭취량/전체주류소비량)
