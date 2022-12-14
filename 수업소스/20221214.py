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
 
#alcohol_rate : 알콜비율 (알콜섭취량/전체주류소비량) 추가
drinks["alcohol_rate"] = \
    drinks["total_litres_of_pure_alcohol"]/drinks["total_servings"]
    
drinks.info()  
#alcohol_rate 컬럼에 결측값 존재.
#전체주류소비량이 0인 경우 불능 => 결측값
#alcohol_rate 컬럼의 값이 결측값인 레코드 조회하기
drinks[drinks["alcohol_rate"].isnull()][["country","total_servings"]]
#alcohol_rate 컬럼의 결측값을 0을 치환하기
drinks["alcohol_rate"]=drinks["alcohol_rate"].fillna(0)
drinks.info()  
#alcohol_rate의 값으로 내림차순 정렬하기. alcohol_rate_rank 저장
alcohol_rate_rank = \
    drinks.sort_values(by="alcohol_rate",ascending=False)\
        [["country","alcohol_rate"]]
alcohol_rate_rank.head()
#대한민국의 순번 출력하기
alcohol_rate_rank.shape
alcohol_rate_rank.country.tolist().index("South Korea")
alcohol_rate_rank.head(15)
#시각화하기
import numpy as np
import matplotlib.pyplot as plt
plt.rc("font",family="Malgun Gothic")
#국가명목록
country_list = alcohol_rate_rank.country.tolist()
#x축값.
x_pos = np.arange(len(country_list))
#막대그래프의 y축값
rank = alcohol_rate_rank.alcohol_rate.tolist()
#막대그래프 
# bar_list : 막대 목록
bar_list = plt.bar(x_pos, rank)
#대한민국 막대의 색을 red로 변경
bar_list[country_list.index("South Korea")].set_color('r')
plt.ylabel('alcohol rate')
plt.title('liquor drink rank by contry')
plt.axis([0, 200, 0, 0.3]) #xy축범위:[x축값의시작,x축값의종료,y축값의시작,y축값의종료]
# korea_rank : 대한민국의 인덱스 순서. 14
korea_rank = country_list.index("South Korea")
#대한민국의 알콜비율데이터 
korea_alc_rate = alcohol_rate_rank\
[alcohol_rate_rank['country'] == 'South Korea']\
    ['alcohol_rate'].values[0]
korea_alc_rate    
#annotate : 그래프에 설명선 추가
plt.annotate('South Korea : ' + str(korea_rank + 1)+"번째", #설명문
            xy=(korea_rank, korea_alc_rate), #x,y 축의 값          
            xytext=(korea_rank + 10, korea_alc_rate + 0.05),#설명문 시작위치
            arrowprops=dict(facecolor='red', shrink=0.05)) #화살표 설정(색상:빨강, 길이:)
plt.show()

'''
total_servings 전체 술소비량을 막대그래프로 작성하고,
 대한민국의 위치를 빨강색으로 표시하기
 1.total_serving_rank = drinks[["country","total_servings"]]
 2.total_serving_rank total_servings 값의 내림차순으로 정렬 
 3.total_serving_rank 데이터를 막대그래프로 작성.
   대한민국의 데이터는 빨강색으로 변경
 4.막대그래프에 설명선 추가하기  
'''
#1.total_serving_rank = drinks[["country","total_servings"]]
total_serving_rank = drinks[['country', 'total_servings']]
total_serving_rank
#2.total_serving_rank total_servings 값의 내림차순으로 정렬 
total_serving_rank = total_serving_rank.sort_values\
                   (by=['total_servings'], ascending=False)
total_serving_rank.head()                   
#국가명 조회
country_list = total_serving_rank.country.tolist()
x_pos = np.arange(len(country_list))
#y축 데이터
rank = total_serving_rank.total_servings.tolist()
#막대그래프
bar_list = plt.bar(x_pos, rank)
# 대한민국의 인덱스 순서
korea_rank = country_list.index("South Korea")
korea_rank
bar_list[korea_rank].set_color('r')
plt.ylabel('total servings')
plt.title('drink servings rank by country')
plt.axis([0, 200, 0, 700])
#korea_serving_rate : 대한민국 전체 술소비량 데이터. y축값
korea_serving_rate = total_serving_rank\
[ total_serving_rank['country']=='South Korea']\
    ['total_servings'].values[0]
korea_serving_rate    
plt.annotate('South Korea : ' + str(korea_rank + 1)+"번째", 
          xy=(korea_rank, korea_serving_rate), 
          xytext=(korea_rank + 10, korea_serving_rate + 50),
          arrowprops=dict(facecolor='red', shrink=50))
plt.show()


'''
서울시 각 구별 CCTV수를 파악하고, 
               인구대비 CCTV 비율을 파악해서 순위 비교
서울시 각 구별 CCTV수 : 01. CCTV_in_Seoul.csv
서울시 인구 현황      : 01. population_in_Seoul.xls
'''
import pandas as pd
CCTV_Seoul = pd.read_csv("data/01. CCTV_in_Seoul.csv")
CCTV_Seoul.info()

Pop_Seoul =  pd.read_excel("data/01. population_in_Seoul.xls")
Pop_Seoul.info()
Pop_Seoul.head()
'''
   header정보 : 3번째행
   셀데이터 : B,D,G,J,N 
'''
Pop_Seoul =  pd.read_excel("data/01. population_in_Seoul.xls",\
                      header=2,usecols="B,D,G,J,N")
Pop_Seoul.info()
Pop_Seoul.head()
#컬럼명 변경하기
CCTV_Seoul.columns
#CCTV_Seoul : 기관명 => 구별
CCTV_Seoul.rename(columns={"기관명":"구별"},inplace=True)
Pop_Seoul.columns
#Pop_Seoul : [구별,인구수,한국인,외국인,고령자]
Pop_Seoul.columns = ["구별","인구수","한국인","외국인","고령자"]
CCTV_Seoul.info()
Pop_Seoul.info()
CCTV_Seoul.head()
Pop_Seoul.head()
#인구데이터의 첫번째 행을 제거하기
Pop_Seoul.drop([0],axis=0,inplace=True)
Pop_Seoul.head()
'''
  CCTV 최근증가율이 높은 구 5개를 조회하기
  1. 최근증가율 컬럼 추가
   (2014~2016까지의 최근 3년간 CCTV수의 합)/(2013년도 CCTV수) * 100
  2. 최근증가율 컬럼으로 내림차순 정렬하여 상위5개만 조회.   
'''
CCTV_Seoul["최근증가율"] =(CCTV_Seoul["2014년"]+CCTV_Seoul["2015년"]\
   +CCTV_Seoul["2016년"])/CCTV_Seoul["2013년도 이전"] *100    
CCTV_Seoul["최근증가율"]    
CCTV_Seoul.sort_values(by="최근증가율",ascending=False)[:5]
'''
  외국인비율,고령자비율이 높은 구 5개 조회하기
  1. Pop_Seoul 데이터의 외국인비율,고령자비율 컬럼 추가하기
    외국인비율 : 외국인/인구수 * 100
    고령자비율 : 고령자/인구수 * 100
  2. 외국인비율,고령자비율로 내리차순 정렬하여 상위5개 조회하기  
'''
Pop_Seoul["외국인비율"]=Pop_Seoul["외국인"]/Pop_Seoul["인구수"] * 100
Pop_Seoul["고령자비율"]=Pop_Seoul["고령자"]/Pop_Seoul["인구수"] * 100
Pop_Seoul.info()
Pop_Seoul.sort_values(by="외국인비율",ascending=False)[:5]
Pop_Seoul.sort_values(by="고령자비율",ascending=False)[:5]

#구별컬럼을 연결컬럼으로 하여 CCTV_Seoul, Pop_Seoul 데이터 합하기
data_result = pd.merge(CCTV_Seoul,Pop_Seoul,on="구별")
data_result.info()
# data_result : 2013년도 이전,2014년,2015년,2016년 컬럼제거
# del 명령사용
del data_result["2013년도 이전"],data_result["2014년"],\
    data_result["2015년"],data_result["2016년"]
data_result.info()
#구별 컬럼을 인덱스로 변경하기 : set_index()
data_result = data_result.set_index("구별")
data_result.info()
#인구수와 소계 컬럼과의 상관계수 구하기
data_result[["인구수","소계"]].corr()
#산점도 그래프 출력하기
import seaborn as sns
sns.pairplot(data_result[["인구수","소계"]])
#회귀 그래프 출력하기
sns.regplot(x="인구수",y="소계", data=data_result)
'''
CCTV비율 컬럼 추가하기 
CCTV비율 : 인구수대비 CCTV갯수  CCTV갯수 / 인구수 * 100
'''
data_result["CCTV비율"]=\
    data_result["소계"]/data_result["인구수"]*100
data_result["CCTV비율"]    
#수평막대그래프로 작성하기
data_result["CCTV비율"].sort_values().plot(kind="barh",
                         grid=True,figsize=(10,10))    
#matplot을 이용하여 산점도와 회귀선 그래프 작성하기
import numpy as np
'''
np.polyfit(x값,y값,차수) : x,y 모든 점과의 차이가 가장
       적은 직선(곡선)의 값일 리턴.

차수 : 1차함수 : 직선 : ax+b (a:기울기, b:y절편)
       2차함수 :곡선  : ax**2 + bx + c
'''
fp1 = np.polyfit\
    (data_result['인구수'], data_result['소계'], 1)
fp1    
f1 = np.poly1d(fp1) #함수.fp1상수값에 맞는 함수
#x의 값. 10만~70만 값을 100개로 균등분할 숫자들
fx = np.linspace(100000, 700000, 100)
fx

plt.figure(figsize=(6,6))
#산점도
plt.scatter(data_result['인구수'], \
            data_result['소계'], s=50)
#회귀선.    
#ls='dashed' : 긴점선
#lw=3 : 선의굵기
plt.plot(fx, f1(fx), ls='dashed', lw=3, color='g')
plt.xlabel('인구수')
plt.ylabel('CCTV')
plt.grid()
plt.show()

#기대cctv갯수=f1(인구수)
# 오차컬럼 : |실제cctv갯수 - 기대cctv갯수| 절대값
# np.abs : 절대값
# data_result["소계"] : 실제cctv갯수
# f1(data_result["인구수"]) : 기대cctv갯수
data_result["오차"] = \
    np.abs(data_result["소계"] - f1(data_result["인구수"]))
data_result["오차"].head()
df_sort = data_result.sort_values(by="오차",ascending=False)
df_sort.head()
plt.figure(figsize=(10,10))  #그래프 크기 지정
#산점도 그래프
plt.scatter(data_result['인구수'], data_result['소계'], 
            c=data_result['오차'], s=50 )
#회귀선 그래프
plt.plot(fx, f1(fx), ls='dashed', lw=3, color='g') #회귀선
#plt.text(x좌표,y좌표,출력문자열,글자크기) : 문자 출력
#df_sort['인구수'][n]*1.02 : x축의 좌표.
#df_sort['소계'][n]*0.98 : y축의 좌표
for n in range(10): #10개의 구정보만 출력.
  plt.text(df_sort['인구수'][n]*1.02, df_sort['소계'][n]*0.98,
         df_sort.index[n], fontsize=10)
plt.xlabel('인구수')
plt.ylabel('CCTV갯수')
plt.colorbar()
plt.grid()
plt.show()

#서울시 경찰서별 범죄율 데이터와 경찰서 위치 데이터 읽기
import numpy as np
import pandas as pd
crime_Seoul = pd.read_csv('data/02. crime_in_Seoul.csv',
                          thousands=',',encoding="cp949")
crime_Seoul.info()
police_state = pd.read_csv("data/경찰관서 위치.csv",encoding="cp949")
police_state.info()
#지방청 컬럼의 내용조회
police_state["지방청"].unique()
#police_Seoul 데이터에 서울청 데이터만 저장하기
police_Seoul = police_state[police_state["지방청"]=='서울청']
police_Seoul.info()
police_Seoul = police_state.groupby("지방청").get_group("서울청")
police_Seoul.info()
police_Seoul["지방청"].unique()
#police_Seoul 데이터에 경찰서 컬럼 값의 종류 출력하기
police_Seoul["경찰서"].unique()
crime_Seoul["관서명"].unique()
#police_Seoul 데이터의 경찰서 컬럼의 내용을 
# XX서로 이름변경하여 관서명 컬럼으로 생성하기
police_Seoul["관서명"]=\
  police_Seoul["경찰서"].apply(lambda x:str(x[2:] + "서"))
police_Seoul["관서명"].unique()
crime_Seoul["관서명"].unique()
police_Seoul.head(10)
#1. police_Seoul 데이터에 
#  지방청, 경찰서,구분 컬럼 제거하기
del police_Seoul["지방청"],police_Seoul["경찰서"],police_Seoul["구분"]
police_Seoul.info()
#2. police_Seoul["관서명"] 중복행을 제거하기
# drop_duplicates() 함수 사용
police_Seoul.head(10)
police_Seoul = \
    police_Seoul.drop_duplicates(subset=["관서명"])
police_Seoul.info()
# police_Seoul 데이터의 주소 컬럼을 이용하여 구별 컬럼을 생성하기
police_Seoul["구별"]=\
    police_Seoul["주소"].apply(lambda x : str(x).split()[1])
police_Seoul["구별"]    
# police_Seoul 데이터의 주소 컬럼제거하기
del police_Seoul["주소"]
police_Seoul.info()
#관서명을 연결컬럼으로 crime_Seoul, police_Seoul 데이터 병합하기
data_result = pd.merge(crime_Seoul,police_Seoul,on="관서명")
data_result.info()
#구별 범죄의 합계를 출력하기
crime_sum = data_result.groupby("구별").sum()
crime_sum

#범죄 종류(강간,강도,살인,절도,폭력)별 검거률 컬럼 추가하기
# 검거율 = 검거/발생 * 100
col_list=['강간', '강도', '살인', '절도', '폭력']
for col in col_list:
    crime_sum[col+"검거율"] =\
        crime_sum[col+" 검거"]/crime_sum[col+" 발생"] * 100
    print(crime_sum[col+"검거율"])
crime_sum.info()    
#검거율 데이터 중 100보다 큰값은 100으로 변경하기
for col in col_list:
   crime_sum.loc[crime_sum[col+"검거율"]>100,col+"검거율"]=100

for col in col_list:
  print(crime_sum.loc[crime_sum[col+"검거율"]>=100,col+"검거율"])
print(crime_sum["절도검거율"])

#구별 절도검거율을 수평막대그래프로 출력하기
# 절도 검거율이 높은 구 부터 그래프로 작성하기
crime_sum["절도검거율"].sort_values().plot\
    (kind="barh",grid=True,figsize=(8,8))
    
#경찰서별 절도검거율을 수평막대그래프로 출력하기
# 절도 검거율이 높은 구 부터 그래프로 작성하기
crime_Seoul.info()  
#1. 절도검거율  컬럼 생성
crime_Seoul["절도검거율"]=\
    crime_Seoul["절도 검거"]/crime_Seoul["절도 발생"] * 100
crime_Seoul["절도검거율"]
crime_Seoul.set_index("관서명",inplace=True)
crime_Seoul["절도검거율"].sort_values().plot\
    (kind="barh",grid=True,figsize=(8,8))
    
    
police_state = \
   pd.read_csv("data/경찰관서 위치.csv",encoding="cp949")
    