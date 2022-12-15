# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 08:42:56 2022

@author: KITCOOP

20221215.py
"""

import pandas as pd
import numpy as np
#서울 구별 CCTV 정보 데이터 읽기
CCTV_Seoul = pd.read_csv("data/01. CCTV_in_Seoul.csv")
CCTV_Seoul.info()

#서울 경찰서별 범죄율 정보 데이터 읽기
crime_Seoul = pd.read_csv('data/02. crime_in_Seoul.csv',
                          thousands=',', encoding='cp949')
crime_Seoul.info()
#전국 경찰서 위치 데이터 읽기
police_state = pd.read_csv('data/경찰관서 위치.csv', encoding='cp949')
police_state.info()
#서울지역 경찰서 위치데이터만 저장
police_Seoul = police_state[police_state["지방청"]=='서울청']

#police_Seoul 데이터의 경찰서 컬럼의 내용을 XX서로 이름변경하여
# 관서명 컬럼으로 생성하기
#1
police_Seoul["관서명"]=\
    police_Seoul["경찰서"].apply((lambda x : str(x[2:]+'서' )))
#1    
police_Seoul["관서명"]=police_Seoul["경찰서"].str[2:]+"서"
police_Seoul["관서명"]

#1. police_Seoul 데이터에 지방청, 경찰서,구분 컬럼 제거하기
del police_Seoul["지방청"],police_Seoul["경찰서"],police_Seoul["구분"]
police_Seoul.info()
#2. police_Seoul["관서명"] 중복행을 제거하기
police_Seoul = police_Seoul.drop_duplicates(subset=["관서명"])
police_Seoul
# police_Seoul 데이터의 주소 컬럼을 이용하여 구별 컬럼을 생성하기
police_Seoul["구별"]= police_Seoul["주소"].apply\
                                  (lambda x : str(x).split()[1])
police_Seoul["구별"]
police_Seoul["구별"]= police_Seoul["주소"].str.split().str[1]
police_Seoul["구별"]
# police_Seoul 데이터의 주소 컬럼제거하기
del police_Seoul["주소"]
police_Seoul.info()
# 관서명 컬럼을 연결컬럼으로 crime_Seoul 데이터와 police_Seoul 데이터를 병합하기 
# data_result 데이터에 저장하기
data_result = pd.merge(crime_Seoul,police_Seoul,on="관서명")
data_result.head()
#구별 범죄의 합계를 출력하기
crime_sum = data_result.groupby("구별").sum()
crime_sum
#범죄 종류(강간,강도,살인,절도,폭력)별 검거율 컬럼 추가하기
# 검거율 = 검거/발생 * 100
col_list=['강간', '강도', '살인', '절도', '폭력']
for col in col_list:
    crime_sum[col+"검거율"] =\
        crime_sum[col+" 검거"]/crime_sum[col+" 발생"] * 100
crime_sum.info()
#검거율 데이터 중 100보다 큰값은 100으로 변경하기
for col in col_list:
    crime_sum.loc[crime_sum[col+"검거율"]>100,col+"검거율"]=100

#확인 출력
for col in col_list:
    print(crime_sum.loc[crime_sum[col+"검거율"]>=100,col+"검거율"])
#구별 범죄데이터
crime_sum

#구별 검거율과, CCTV 갯수를 산점도와 회귀선으로 출력하기.
# 오차가 큰 10개 구이름을 그래프로 출력하기
#crime_sum 인덱스를 구별 컬럼으로 변경하기
crime_sum = crime_sum.reset_index()
crime_sum
#구별 컬럼으로 CCTV_Seoul,crime_sum 데이터를 병합하여 data_result에 저장하기
CCTV_Seoul.info()

#기관명 컬럼을 구별 컬럼명 변경
CCTV_Seoul.rename(columns={"기관명":"구별"},inplace=True)
CCTV_Seoul.drop\
    (["2013년도 이전","2014년","2015년","2016년"],axis=1,inplace=True)
#cctv데이터+crime 데이터를 구별로 병합    
data_result = pd.merge(CCTV_Seoul,crime_sum, on="구별")
data_result.info()

#절도검거율과cctv 회귀선과 산점도 출력하기
fp1 = np.polyfit(data_result['소계'], data_result['절도검거율'], 2) #상수값
f1=np.poly1d(fp1) #회귀선을 위한 함수
fx=np.linspace(500,4000,100)
#data_result 데이터에 오차 컬럼을 추가하기
# 실제 검거율과 기대검거율의 차이의 절대값 저장
data_result["오차"]=np.abs(data_result["절도검거율"]-f1(data_result['소계']))
#오차의 내림차순으로 정렬하여 df_sort 데이터 저장
df_sort = data_result.sort_values(by="오차", ascending=False)
df_sort
import matplotlib.pyplot as plt 
plt.rc('font', family ='Malgun Gothic')
plt.figure(figsize=(14,10))
plt.scatter(df_sort['소계'],df_sort["절도검거율"],c=df_sort['오차'], s=50)
plt.plot(fx, f1(fx), ls='dashed', lw=3, color='g')
for n in range(10):
    plt.text(df_sort.iloc[n,]["소계"]*1.001,  #x축값
             df_sort.iloc[n,]['절도검거율']*0.999, #y축값
             df_sort.iloc[n,]['구별'], fontsize=10)
plt.xlabel('CCTV 갯수')
plt.ylabel('절도범죄 검거율')
plt.title("CCTV와 절도 검거율 분석")
plt.colorbar()
plt.grid()
plt.show()

'''
경찰서별 범죄발생건수, CCTV 갯수를 산점도와 회귀선으로 출력하기.
   단 CCTV의 갯수는 구별로 지정한다.
   발생컬럼 : 모든 범죄발생건수 합.
         강도발생+강간발생+절도발생+
'''
crime_Seoul = pd.read_csv('data/02. crime_in_Seoul.csv',
                          thousands=',', encoding='cp949')

crime_Seoul.info()
crime_Seoul["발생"]=crime_Seoul["살인 발생"] + \
                    crime_Seoul["강도 발생"] + \
                    crime_Seoul["강간 발생"] + \
                    crime_Seoul["절도 발생"] + \
                    crime_Seoul["폭력 발생"] 
crime_Seoul["발생"]                    
crime_Seoul = pd.merge(crime_Seoul,police_Seoul, on="관서명")
crime_Seoul.info()
crime_Seoul[["관서명","구별"]]
#cctv,crime 데이터 병합
data_result = pd.merge(CCTV_Seoul,crime_Seoul,on="구별")
data_result.info()
fp1=np.polyfit(data_result["소계"],data_result["발생"],1)
fp1
fx=np.linspace(500,3000,100)
f1=np.poly1d(fp1)
data_result["오차"]=np.abs(data_result["발생"]-f1(data_result["소계"]))
data_result["오차"]
#오차데이터기준 내림차순 정렬
df_sort = data_result.sort_values(by="오차",ascending=False)
df_sort

plt.figure(figsize=(7,5))
plt.scatter(df_sort['소계'],df_sort["발생"],c=df_sort["오차"],s=50)
plt.plot(fx,f1(fx),ls="dashed",lw=3,color='g') #회귀선그래프
for n in range(10):
    plt.text(df_sort.iloc[n,]['소계']*1.02, df_sort.iloc[n,]['발생']*0.997, 
             df_sort.iloc[n,]["관서명"], fontsize=15)
plt.xlabel('CCTV 갯수')
plt.ylabel('범죄발생건수')
plt.title('범죄발생과 CCTV 분석')
plt.colorbar()
plt.grid()
plt.show()

########################################
#  머신러닝 : 기계학습. 예측. AI(인공지능.)
#             변수(컬럼,피처)들의 관계를 통해서 예측하는 과정
#   지도학습   : 정답을 지정
#             회귀분석 : 가격,주가,매출 예측하는 과정. 
#                       연속성 있는 데이터의 예측에 사용
#             분류    : 데이터의 선택. 평가
#   비지도학습 : 정답이 없음.
#             군집 : 비숫한 데이터들 끼리 그룹화함.
#   강화학습 : 행동을 할때마다 보상을 통해 학습하는 과정 

#  머신러닝 프로세스()
#   데이터정리(전처리) - 데이터분리(훈련/검증/테스트)-알고리즘준비
# ->모형학습(훈련데이터) - 예측(테스트데이터) - 모형평가 -> 모형활용
########################################
'''
   회귀 분석(regression) 
     단순회귀분석 : 독립변수,종속변수가 한개씩
       독립변수(설명변수) : 예측에 사용되는 데이터
       종속변수(예측변수) : 예측해야 하는 데이터
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("data/auto-mpg.csv")
df.info()
#1. horsepower컬럼의 자료형을 float 형으로 변경하기
# ?를 결측값변경, 결측값 행을 삭제.변경
df["horsepower"].unique()
df["horsepower"].replace("?",np.nan,inplace=True)
df.info()
df.dropna(subset=["horsepower"],axis=0,inplace=True)
df.info()
df["horsepower"]=df["horsepower"].astype(float)
df.info()

#머신러닝에 필요한 속성(열,컬럼,변수,피처) 선택하기
ndf=df[['mpg','cylinders','horsepower','weight']]
ndf.corr()
sns.pairplot(ndf)

#독립변수,종속변수 
X = ndf[["weight"]]
Y = ndf["mpg"]
len(X)
len(Y)
#데이터분리(훈련/테스트)
'''
train_test_split : 훈련/테스트 데이터 분리 함수.
train_test_split(독립변수,종속변수,테스트데이터비율,seed설정)
test_size=0.3 : 훈련:테스트=7:3 기본값 : 0.25 
seed설정 : 데이터의 복원을 위한 설정
'''
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = \
    train_test_split(X,Y,test_size=0.3,random_state=10)
len(X_train) #392*0.7=274.4 => 274 : 훈련데이터 독립변수
len(X_test)  #392*0.3=117.6 => 118 : 테스트데이터 독립변수
len(Y_train)                      #  훈련데이터 종속변수
len(Y_test)                       #  테스트데이터 종속변수

#알고리즘 준비 : 선형회귀분석:LinearRegression
from sklearn.linear_model import LinearRegression
lr = LinearRegression()

#모형학습
lr.fit(X_train,Y_train)

#예측
y_hat = lr.predict(X_test)

#평가
y_hat[:10] #예측데이터
Y_test[:10] #실제데이터
r_square = lr.score(X_test,Y_test)
r_square  #결정계수. 값이 1에 가까울 수록 성능이 좋다. 0.6530296487379842
r_square = lr.score(X,Y)
r_square  #0.6925517573534928

#전체 데이터 평가하기
y_hat = lr.predict(X) #전체 데이터 예측 
#y_hat : 모형에서 예측된 데이터
#Y     : 실제 데이터 
plt.figure(figsize=(10,5))
#kdeplot : 값의 밀도
ax1=sns.kdeplot(Y,label="Y") #실제데이터 밀도
ax2=sns.kdeplot(y_hat,label="y_hat",ax=ax1) #예측데이터 밀도
plt.legend()
plt.show()

# 알고리즘 선택 : PolynomialFeatures. 
# LinearRegression : 선형회귀분석 : ax+b
# PolynomialFeatures : 다항회귀분석 : ax**2+bx+c

from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(2) #2차항.
X_train.shape
X_train.iloc[0]
X_train_poly=poly.fit_transform(X_train) #다항식에 처리가능한 데이터 변환
X_train_poly.shape
X_train_poly[0]

pr = LinearRegression()
pr.fit(X_train_poly, Y_train) #모형 학습
#평가데이터도 다항식 데이터 변환
X_poly = poly.fit_transform(X) 
y_hat = pr.predict(X_poly) #예측
plt.figure(figsize=(10, 5))
ax1 = sns.kdeplot(Y, label="Y")
ax2 = sns.kdeplot(y_hat, label="y_hat", ax=ax1)
plt.legend()
plt.show()

r_square = pr.score(X_poly,Y)
r_square #0.7151455671417561

#결정계수 : 1 - 잔차제곱합/총변환량
#          1 - u/v
u = ((Y-y_hat)**2).sum()
v = ((Y-Y.mean())**2).sum()
1-(u/v) #0.7151455671417561
'''
  단순회귀분석 : 독립변수,종속변수가 한개인 경우
    단항 : 1차함수
    다항 : 다차원함수
  다중회귀분석 : 독립변수가 여러개, 종속변수는 한개
    Y=a1X1 + a2X2 + ...anXn + b
'''
#독립변수,종속변수 선택
X=ndf[['cylinders','horsepower','weight']] #독립변수
Y=ndf["mpg"] #종속변수 
X.info()
#데이터 훈련/테스트데이터 분리
X_train,X_test,Y_train,Y_test = \
    train_test_split(X,Y,test_size=0.3,random_state=10)

#알고리즘 선택. : 선형회귀분석
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
#학습하기
lr.fit(X_train,Y_train)
#예측하기
y_hat=lr.predict(X)
#평가하기
r_square = lr.score(X,Y) #0.7062412894103278
r_square
plt.figure(figsize=(10, 5))
ax1 = sns.kdeplot(Y, label="Y")
ax2 = sns.kdeplot(y_hat, label="y_hat", ax=ax1)
plt.legend()
plt.show()

#단순 회귀분석의 간단한 예
#독립변수 1개, 종속변수 1개.
x=[[10],[5],[9],[7]]
y=[100,50,88,75]
model=LinearRegression()
model.fit(x,y)
result=model.predict([[7],[8],[4],[6]])
result

#다중 회귀분석의 간단한 예
#독립변수 2개, 종속변수 1개.
x=[[10,3],[5,2],[9,3],[7,2]]
y=[100,50,88,80]
model=LinearRegression()
model.fit(x,y)
result=model.predict([[7,3],[8,2],[4,3],[6,3]])
result

#####################################
#  https://data.kma.go.kr/ : 기후통계분석 > 기온분석 데이터 다운받기
#    1904 ~ 전일까지 : seoul_1215.csv 저장
#   2022-12-15일 날짜 예측하기
##############################################
#seoul_1215.csv 읽기
seoul = pd.read_csv("data/seoul_1215.csv",encoding="cp949")
seoul.info()
seoul.head()
#\t 제거하기
seoul["날짜"]=seoul["날짜"].str.replace("\t","")
seoul.info()
seoul.head()
#년도 컬럼생성
seoul["년도"]=seoul["날짜"].str[:4]
seoul.head()
#월일 컬럼생성
seoul["월일"]=seoul["날짜"].str[5:]
seoul.head()
#seoul1215 변수에 12-15일 날짜만 저장하기
seoul1215=seoul[seoul["월일"]=="12-15"]
seoul1215.info()
seoul1215.tail()
#지점 제거
del seoul1215["지점"]
#컬럼명 변경
seoul1215.columns=\
    ["날짜","평균기온","최저기온","최고기온","년도","월일"]
seoul1215.info()
#최저기온이 결측값인 데이터 조회하기
seoul1215[seoul1215["최저기온"].isnull()]
#최저기온이 결측값인 데이터 제거하기
seoul1215= seoul1215.dropna(subset=["최저기온"],axis=0)
seoul1215.info()
#독립변수,종속변수
X=seoul1215[["년도"]]
Y=seoul1215["최저기온"]
#알고리즘
model=LinearRegression()
model.fit(X,Y)   
result = model.predict([[2022]])
result

#2 다중 회귀분석 : 독립변수 여러기
X=seoul1215[["년도","최고기온"]] ##2022 1
Y=seoul1215["최저기온"]
model=LinearRegression()
model.fit(X,Y)   
result = model.predict([[2022,1]])
result
import matplotlib.pyplot as plt
plt.rc("font",family="Malgun Gothic")
seoul1215 = seoul1215.set_index("년도")
seoul1215.plot()

#각각의 생일일자를 그래프로 작성하기
seoul.info()
seoul0409=seoul[seoul["월일"]=='04-09']
seoul0409.info()
seoul0409 = seoul0409.set_index("년도")
del seoul0409["지점"]
seoul0409.plot()

##################
#  분류 : 지도학습
#   설명변수(독립변수)
#   목표변수(종속변수)
#   알고리즘 : KNN(k-Nearset-Neighbors) 
#             SVM(Support Vector Machine)
#             Decision Tree 
##################
