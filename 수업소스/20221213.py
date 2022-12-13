# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 09:33:15 2022

@author: KITCOOP

20221213.py
"""

#파일 읽기
import pandas as pd
chipo = pd.read_csv("data/chipotle.tsv",sep="\t")
chipo.info()
'''
데이터 속성 설명
order_id : 주문번호
quantity : 아이템의 주문수량
item_name : 주문한 아이템의 이름
choice_description : 주문한 아이템의 상세 선택 옵션
item_price : 주문 아이템의 가격 정보
'''
#chipo 데이터의 행열의 갯수 출력하기
chipo.shape
#컬럼명
chipo.columns
#인덱스명
chipo.index
# order_id 주문번호이므로, 숫자형 분석의 의미가 없다.
# order_id  컬럼의 자료형을 문자열로 변경하기
chipo["order_id"] = chipo["order_id"].astype(str)
chipo.info()
#판매상품명과 상품의 갯수 출력하기
chipo["item_name"].unique()
len(chipo["item_name"].unique())
#item_price 컬럼을 실수형 변경

#1 
chipo["item_price"] = \
chipo["item_price"].str.replace("$","").astype(float)
chipo.info()
#주문금액 합계
chipo["item_price"].sum()

#2 apply 함수 : 요소들에 적용되는 함수
# apply(함수명||람다식)
chipo["item_price"] = \
    chipo["item_price"].apply(lambda x : float(x[1:]))

chipo["item_price"].sum()

#주문당 평균 주문 금액
'''
    주문번호  주문금액
       1       1
       1       3
       2       1
       2       1
       
       전체 주문금액 : 6
       주문건수      : 2
       주문당평균금액 : 3
'''
#총합계금액
hap = chipo["item_price"].sum()
hap
#주문건수
cnt = len(chipo.groupby("order_id"))
cnt
# 주문당평균금액
hap/cnt
# 주문당평균금액
chipo.groupby("order_id")["item_price"].sum().mean()
'''
  한번의 주문시 주문금액이 50달러 이상인 주문의 id 출력하기
  주문번호   주문금액
     1         25
     1         30
     2          2
     2         45
     
     50달러이상인 주문번호 : 1          
'''
#주문당 합계금액
order_id_tot = chipo.groupby("order_id").sum()
order_id_tot
result = order_id_tot[order_id_tot["item_price"] >= 50]
result
len(result)
list(result.index) #50달러 이상 주문한 주문번호
#50달러 이상인 주문 정보를 chipo_50 에 저장하기
chipo_50 = chipo[chipo["order_id"].isin(result.index)]
chipo_50
chipo_50.info()

#50달러 이상인 주문 정보를 출력하기
chipo_51 = chipo.groupby("order_id").\
    filter(lambda x : sum(x["item_price"]) >= 50)
chipo_51.info()

# item_name 별 단가를 조회하기
# item_name 별 최소값
price_one = chipo.groupby("item_name").min()["item_price"]
price_one
# 단가의 분포를 히스토그램으로 출력하기
import matplotlib.pyplot as plt
plt.rc("font",family="Malgun Gothic")
plt.hist(price_one)
plt.ylabel("상품갯수")
plt.title("상품단가 분포")

price_one.plot(kind="hist")
plt.ylabel("상품갯수")
plt.title("상품단가 분포")

#단가가 가장 높은 상품 10개 조회하기
price_one.sort_values(ascending=False)[:10]
#단가가 가장 높은 상품의 이름 조회하기
max_item = price_one.max()
list(price_one[price_one==max_item].index)

#주문당 주문금액이 가장 높은 5건의 주문 총수량을 조회하기
price5 = chipo.groupby("order_id").sum()\
  .sort_values(by="item_price",ascending=False)[:5]
price5  
price5["quantity"].sum() #103
#주문당 주문금액이 높은 5건 주문의, 정보 조회하기
chipo_5 = chipo[chipo["order_id"].isin(price5.index)]\
  [["order_id","item_name","quantity","item_price"]]
chipo_5

# Veggie Salad Bowl 몇번 주문되었는지 출력하기
'''
   주문번호 주문상품
      1     Veggie Salad Bowl
      1     Veggie Salad Bowl
      2     Veggie Salad Bowl
    
    => 2번주문     
'''
chipo_salad = \
    chipo[chipo["item_name"]=='Veggie Salad Bowl']
len(chipo_salad)
len(chipo_salad.groupby("order_id"))
# 한 주문 내에서 중복 집계된 item_name을 제거합니다.
chipo_salad = \
 chipo_salad.drop_duplicates(['item_name', 'order_id']) 
print(len(chipo_salad))

###################
#  전세계 음주 데이터 분석하기 : drinks.csv
import pandas as pd
drinks = pd.read_csv("data/drinks.csv")
drinks.info()
'''
  country : 국가명
  beer_servings : 맥주소비량
  spirit_servings : 음료소비량
  wine_servings : 와인소비량   
  total_litres_of_pure_alcohol : 순수 알콜량
  continent : 대륙명
'''
drinks.head()
# 변수 = 컬럼 = 피처
# 상관계수 : 두연속적인 데이터의 상관관계 수치
# 피어슨 상관계수 : 기본. 
beer_wine_corr=\
    drinks[["beer_servings","wine_servings"]].corr()
beer_wine_corr

beer_wine_corr=drinks[["beer_servings","wine_servings"]]\
    .corr(method="pearson")
beer_wine_corr
# 켄달 상관계수 : 샘플 사이즈가 작은 경우.
#               동률데이터의 확율이 높은 경우
beer_wine_corr=drinks[["beer_servings","wine_servings"]]\
    .corr(method="kendall")
beer_wine_corr
# 스피어만 상관계수 : 정규화가 되지 않는 데이터에 많이 사용
beer_wine_corr=drinks[["beer_servings","wine_servings"]]\
    .corr(method="spearman")
beer_wine_corr
drinks.columns
cols = drinks.columns[1:-1]

corr = drinks[cols].corr()
corr
corr.values

#상관계수 시각화하기
#히트맵을 이용하여 시각화 하기
import seaborn as sns
cols_view = ["beer","spirit","wine","alcohol"]
sns.set(font_scale=1.5) #글자크기.
hm=sns.heatmap(corr.values,  #데이터
                cbar=True,   #색상맵
                annot=True,  #데이터값표시
                square=True, #히트맵을 사각형으로 출력
                yticklabels=cols_view, #y축 표시 라벨
                xticklabels=cols_view) #x축 표시 라벨

#seaborn 모듈의 산점도을 이용하여 시각화 하기
sns.pairplot(drinks[cols])
plt.show()

#회귀그래프 작성하기
sns.regplot\
 (x="beer_servings",y="total_litres_of_pure_alcohol",
  data=drinks)
 
#각 변수의 결측값 갯수 조회하기
drinks.isnull().sum() 
#대륙별 국가수 조회하기
drinks["continent"].value_counts()
drinks["continent"].value_counts(dropna=False)
drinks.groupby("continent").count()["country"]
#continent 컬럼의 결측값을 OT로 변경하기
#fillna : 결측값을 다른 값으로 치환함수
drinks["continent"]=drinks["continent"].fillna("OT")
drinks.info()
# 대륙별 국가의 갯수를 파이그래프로 출력하기
sns.set(font_scale=1)
#tolist() : 리스트로 형변환
labels = drinks['continent'].value_counts().index.tolist()
labels
#'AF', 'EU', 'AS', 'OT', 'OC', 'SA'
explode = (0, 0, 0, 0.1, 0, 0) 
plt.pie(drinks['continent'].value_counts(), #데이터값
    labels=labels,                     #라벨명. 대륙명
    autopct='%.0f%%',  #비율표시. %.0f : 소숫점이하 없음. %%:%문자
    explode=explode,  #파이의 위치지정. 0.1 : 1/10만큼 밖으로 표시
    shadow=True)
plt.title('null data to \'OT\'')

# 대륙별 spirit_servings의 평균, 최소, 최대, 합계를 출력.
drinks.groupby("continent").mean()["spirit_servings"]
drinks.groupby("continent").min()["spirit_servings"]
drinks.groupby("continent").max()["spirit_servings"]
drinks.groupby("continent").sum()["spirit_servings"]

drinks.groupby("continent")["spirit_servings"].agg\
    (["mean","min","max","sum"])

drinks.groupby("continent").agg\
    (["mean","min","max","sum"])["spirit_servings"]

#total_litres_of_pure_alcohol : 알콜량
#대륙별 알콜량의 평균이 전체 알콜량 평균보다 많은 대륙을 조회하기
#전체 알콜 평균
total_mean = drinks["total_litres_of_pure_alcohol"].mean()
total_mean
#대륙별 알콜 평균
cont_mean = \
drinks.groupby("continent").mean()["total_litres_of_pure_alcohol"]
cont_mean
#대륙 조회
list(cont_mean[cont_mean > total_mean].index)

#대륙별 beer_servings 평균이 가장 많은 대륙 조회하기
drinks.groupby("continent").beer_servings.mean().idxmax()
#대륙별 beer_servings 평균이 가장 적은 대륙 조회하기
drinks.groupby("continent").beer_servings.mean().idxmin()

'''
  대륙별 total_litres_of_pure_alcohol 섭취량 평균 시각화 하기
'''
import numpy as np
plt.rc("font",family="Malgun Gothic")
cont_mean = \
    drinks.groupby("continent")["total_litres_of_pure_alcohol"].mean()
cont_mean    
#대륙명 : x축의 라벨
continents = cont_mean.index.tolist()
continents
continents.append("Mean") #x축 라벨 추가 
x_pos = np.arange(len(continents)) #0~6까지 숫자
#y축 데이터 : 대륙별 평균값
alcohol = cont_mean.tolist()
alcohol
alcohol.append(total_mean) #전체 알콜섭취 평균
#그래프화
#plt.bar : 막대그래프
#bar_list : 막대그래프의 막대 목록
bar_list =\
plt.bar(x_pos, alcohol, align='center',alpha=0.5)
bar_list
#bar_list[len(continents) - 1] : bar_list[6] 막대
#set_color('r') : 색상 설정. r:red 
bar_list[len(continents) - 1].set_color('r') #7번째 막대색 red로 설정
#plt.plot : 선그래프
# [0., 6] : x축 값
# [total_mean, total_mean] : y축 값
# "k-" : 검정 실선. --:점선
plt.plot([0., 6], [total_mean, total_mean], "k-")
#x축의값을 변경. : 0~6 숫자를 continents의 내용 변경
plt.xticks(x_pos, continents)
plt.ylabel('total_litres_of_pure_alcohol') #y축설명
plt.title('대륙별 평균알콜 섭취랑') #제목
plt.show()

'''
대륙별 beer_servings의 평균를 막대그래프로 시각화
가장 많은 맥주를 소비하는 대륙(EU)의 막대의 색상을 빨강색("r")으로 변경하기 
전체 맥주 소비량 평균을 구해서 막대그래프에 추가
평균선을 출력하기. 평균 막대 색상은 노랑색 ("y")
평균선은 검정색("k--")
'''
# 전체 맥주 소비량 평균
total_beer = drinks.beer_servings.mean()
total_beer
# 대륙별 맥주 소비량 평균
beer_mean = \
 drinks.groupby("continent").beer_servings.mean()
beer_mean

continents = beer_mean.index.tolist()
continents
continents.append("Mean")
continents

x_pos = np.arange(len(continents))
x_pos
#beer_mean 의 최대값
beer_mean.max()
#beer_mean의 최대인덱스
beer_mean.idxmax()
#beer_mean의 최대인덱스 순번. 0부터 시작
beer_mean.argmax()
#continents Mean 데이터의 인덱스순번 
continents.index("Mean")

beer = beer_mean.tolist()
beer
beer.append(total_beer)
beer
bar_list = plt.bar(x_pos,beer,align="center",alpha=0.5)
bar_list[beer_mean.argmax()].set_color("r")
bar_list[continents.index("Mean")].set_color("y")
plt.plot([0,6],[total_beer,total_beer],'k--')
plt.ylabel("맥주소비량")
plt.title("대륙별 평균 맥주 소비량")
plt.xticks(x_pos,continents)

#################################
#대한민국은 얼마나 술을 독하게 마시는 나라인가?

drinks["total_servings"] =\
    drinks["beer_servings"] + \
    drinks["spirit_servings"] +\
    drinks["wine_servings"]
 


