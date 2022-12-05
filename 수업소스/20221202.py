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
import pandas as pd
dict_data = {'c0':[1,2,3], 'c1':[4,5,6], 'c2':[7,8,9], \
             'c3':[10,11,12], 'c4':[13,14,15]}
df = pd.DataFrame(dict_data, index=['r0', 'r1', 'r2'])
df

#인덱스 r0,r1,r2,r3,r4 증가
#df.index=['r0','r1','r2','r3','r4'] #오류 발생. 행의 갯수틀림
#reindex() 함수 : 인덱스 재설정. 행의 추가.
ndf = df.reindex(['r0','r1','r2','r3','r4'])
ndf
#NaN :결측값. (값이없다)
#fill_value=0 : 추가된 내용에 0 값으로 채움
ndf2 = df.reindex(['r0','r1','r2','r3','r4'],fill_value=0)
ndf2

#sort_index()  : 인덱스명으로 정렬
#sort_values() : 기준컬럼의 값으로 정렬
print(df.sort_index()) #인덱스명으로 오름차순정렬
print(df.sort_index(ascending=False)) #인덱스명으로 내림차순정렬
print(ndf2.sort_index(ascending=False)) #인덱스명으로 내림차순정렬
#c1 컬럼의 값을 기준으로 내림차순정렬
print(df.sort_values(by="c1",ascending=False))   

exam_data={"이름":["서준","우현","인아"],
           "수학":[90,80,70],
           "영어":[98,89,95],
           "음악":[85,95,100],
           "체육":[100,90,90]}
df=pd.DataFrame(exam_data)
df
#1. 이름 컬럼을 인덱스 설정하기
df.set_index("이름",inplace=True)
df
#이름의 역순으로 정렬하기
df.sort_index(ascending=False,inplace=True)
df
#영어점수의 역순으로 정렬하기
df.info()
df.sort_values(by="영어",ascending=False,inplace=True)
df
df.sort_values(by="음악",ascending=False,inplace=True)
df
#총점 컬럼 생성하여, 총점의 역순으로 출력하기
df["총점"]=df["수학"]+df["영어"]+df["음악"]+df["체육"]
df
df.sort_values(by="총점",ascending=False,inplace=True)
df

##################################################
# titanic 데이터셋 연습
# seaborn 모듈에 저장된 데이터
'''
survived	생존여부
pclass	좌석등급 (숫자)
sex	성별 (male, female)
age	나이
sibsp	형제자매 + 배우자 인원수
parch: 	부모 + 자식 인원수
fare: 	요금
embarked	탑승 항구
class	좌석등급 (영문)
who	성별 (man, woman)
adult_male 성인남자여부 
deck	선실 고유 번호 가장 앞자리 알파벳
embark_town	탑승 항구 (영문)
alive	생존여부 (영문)
alone	혼자인지 여부
'''

import pandas as pd
import seaborn as sns #시각화모듈
#seaborn 모듈에 저장된 데이터셋 목록
print(sns.get_dataset_names())
#titanic데이터 로드. 
titanic = sns.load_dataset("titanic")
titanic.info()
#
titanic.head()
titanic
#pclass,class 데이터만 조회하기
titanic[["pclass","class"]].head()

#컬럼별 건수 조회하기
titanic.count() #결측값을 제외한 데이터
# 건수 중 가장 작은 값 조회하기
titanic.count().min()
# 건수 중 가장 작은 값의 인덱스 조회하기
titanic.count().idxmin()
type(titanic.count())

#titanic의 age,fare 컬럼만을 df 데이터셋에 저장하기
df = titanic[["age","fare"]]
df.info()
#df 데이터의 평균 데이터 조회
df.mean()
#df 데이터의 최대나이와 최소나이 조회
df.age.max()
df.age.min()
#나이별 인원수를 조회. 최대 인원수를 가진 5개의 나이 조회
#값의 갯수. 내림차순 정렬하여 조회 
df.age.value_counts().head()
#인원수가 많은 나이 10개 조회 
df.age.value_counts().head(10)

#승객 중 최고령자의 정보 조회하기
df[df["age"]==df["age"].max()]
#1
titanic[titanic["age"]==titanic["age"].max()]
#2
titanic["age"].idxmax()
titanic.iloc[titanic["age"].idxmax()]

# 데이터에서 생존건수(342), 사망건수(549) 조회하기
titanic.columns
titanic["survived"].value_counts()
titanic["alive"].value_counts()

#성별로 인원수 조회하기
titanic["sex"].value_counts()
titanic["who"].value_counts()

#성별로 생존건수 조회하기
cnt=  titanic[["sex","survived"]].value_counts()
cnt
type(cnt)
cnt.index

#컬럼 : 변수,피처 용어사용.
# 상관 계수 : -1 ~ 1사이의 값. 변수의 상관관계 수치로 표현
titanic.corr()
titanic[["survived","age"]].corr()
'''
   빅데이터 특징
   3V
   1. Volume(규모) : 데이터의 양이 대용량
   2. Velocity(속도) : 데이터의 처리 속도가 빨라야 한다.
   3. Variety(다양성) : 데이터의 형태가 다양함.
      정형데이터:데이터베이스, csv
      반정형데이터 : json,xml,html
      비정형데이터 : 이미지, 음성 
'''
#1. seaborn 데이터에서 mpg 데이터 로드하기
import seaborn as sns
mpg = sns.load_dataset("mpg")
mpg.info()
'''
mpg : 연비
cylinders : 실린더 수
displacement : 배기량
horsepower : 출력
weight : 차량무게
acceleration : 가속능력
model_year : 출시년도
origin : 제조국
name : 모델명
'''
#2. 제조국별 자동차 건수 조회하기
mpg.origin.value_counts()
mpg['origin'].value_counts()

#3. 제조국 컬럼의 값을 조회하기. 
# unique() : 중복을 제거하여 조회 
mpg.origin.unique()  #[usa, japan,europe]

#4. 출시년도의 데이터 조회하기
mpg.model_year.unique()
mpg.model_year.value_counts()
#5. mpg 데이터의 통계정보 조회하기
mpg.describe() #숫자형태의 데이터만 조회
mpg.describe(include="all") #모든 데이터 조회

type(mpg)
#mpg데이터의 행의값,열의값 조회
mpg.shape #(398,9)튜플데이터 : 398행 9열
#행의값 조회
mpg.shape[0]
#열의값 조회
mpg.shape[1]

#모든 컬럼의 자료형을 조회하기
mpg.dtypes

#mpg 컬럼의 자료형을 조회하기
mpg["mpg"].dtypes

#6. mpg. 데이터의 mpg,weight 컬럼의 최대값 조회하기
mpg.mpg.max()
mpg.weight.max()
mpg[["mpg","weight"]].max()
#7. mpg. 데이터의 mpg,weight 컬럼의 기술통계 정보 조회하기
mpg[["mpg","weight"]].describe(include="all")

#8. 최대 연비를 가진 자동차의 정보 조회하기
mpg[mpg["mpg"]==mpg["mpg"].max()]
mpg.loc[mpg["mpg"]==mpg["mpg"].max()]
mpg.iloc[mpg["mpg"].idxmax()]
'''
  상관계수 : -1 ~ 1 사이의 값을 가짐
      컬럼값의 관계를 수치로 표현
      1 : 상관도 일치. c1 1증가, c2 1 증가 
      -1 : 상관도 반비례. c1 1증가, c2 -1 증가 
      0 : 상관이 없다로 판단
'''
#mpg 데이터의 컬럼간의 상관계수 조회하기
mpg.corr()
#mpg mpg, weight 데이터의 컬럼간의 상관계수 조회하기
mpg[["mpg","weight"]].corr()

#시각화 하기
#연비와 차량의 무게의 관계를 시각화하기
'''
  산점도 : 두개 컬럼의 각각의 값들을 x,y축에 점으로 표현
          값의 분포를 알수 있다.
          컬럼사이의 관계를 시각화 한다
          kind="scatter" : 산점도
'''
mpg.plot(x="mpg",y="weight", kind="scatter")
# 히스토그램 : 데이터의 빈도수 시각화
#             kind="hist"
mpg.mpg.plot(kind="hist")

# 남북한발전전력량.xlsx 파일을 읽어 df에 저장하기
#첫번째 sheet. sheet가 한개만 있음.
df = pd.read_excel("data/남북한발전전력량.xlsx")
df
df.info()
df.head()
df.tail()
#0,5행데이터의 2열 이후의 정보만 ndf에 저장하기
ndf = df.iloc[[0,5]]
ndf.info()
ndf.head()
ndf = df.iloc[[0,5],2:]
ndf.info()
#선그래프 출력하기
ndf.plot() #컬럼별 선 그래프작성 
ndf.info()
#남한,북한별로 그래프 작성필요 : 남한,북한 컬럼으로, 
# 행열을 바꿔야함. 행=>열, 열=>행
# 전치행렬 : 행과 열이 바뀌는 행렬. ndf.T
ndf2 = ndf.T
ndf2
ndf2.info()
#컬럼명 변경하기
ndf2.columns=["South","North"]
ndf2.info()
ndf2
#선그래프로 출력하기
ndf2.plot() #범례:컬럼명
#막대그래프로 출력하기
ndf2.plot(kind="bar") #범례:컬럼명. 컬럼한개:막대한개
ndf2.info()
#데이터 값을 정수형(숫자형)변환
ndf2=ndf2.astype(int)
ndf2.info()
#히스토그램
ndf2.plot(kind="hist") 

### 시도별 전출입 인구수 분석하기 
#1. excel 파일 읽기
import pandas as pd
df=pd.read_excel("data/시도별 전출입 인구수.xlsx")
df.info()
df["전출지별"].head()
#2. 결측값 처리 : 앞데이터 채움 
df = df.fillna(method="ffill")
df.info()
df["전출지별"].head()
df["전출지별"].tail()

#3. 전출지가 서울이고, 전입지는 서울이 아닌 데이터만
#   추출하기. df_seoul에 저장
mask = (df["전출지별"]=='서울특별시') &\
       (df["전입지별"]!='서울특별시')
mask
#mask의 true/false 갯수 구하기
mask.unique()
mask.value_counts()
df_seoul = df[mask]
df_seoul.info()
df_seoul.head()
#4.컬럼명을 전입지별=>전입지 변경하기
df_seoul.rename(columns={"전입지별":"전입지"},inplace=True)
df_seoul.info()
#5.전출지별 컬럼을 제거하기
df_seoul = df_seoul.drop("전출지별",axis=1)
df_seoul.info()
df_seoul.head()
#6. 전입지 컬럼을 인덱스로 설정하기
df_seoul.set_index("전입지",inplace=True)
df_seoul.head()
df_seoul.info()
#7. 경기도로 이동한 데이터만 sr1에 저장하기
sr1 = df_seoul.loc["경기도"] #series객체
sr1
sr1.plot() #선그래프로 작성
#8. 경기도,전국로 이동한 데이터만 df1에 저장하기
df1 = df_seoul.loc[["전국","경기도"]]
df1.info()
df1.T.plot() #선그래프 출력
#matplot 시각화 모듈
import matplotlib.pyplot as plt
#한글이 가능한 폰트로 설정 : 맑은 고딕. 기본폰트:한글불가
plt.rc("font",family="Malgun Gothic")
plt.plot(sr1)
plt.title("서울=>경기 인구 이동")
plt.xlabel("년도")
plt.ylabel("이동인구수")
plt.xticks(rotation="vertical") #x축의 레이블을 세로로 표시
