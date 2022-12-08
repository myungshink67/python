# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 17:05:30 2022

@author: KITCOOP
20221208.py
"""
import numpy as np
np.linspace(1,10,10)
'''
numpy 기본 함수
  np.arange(15) : 0 ~ 14까지의 숫자를 1차원 배열로 생성
  arr.reshape(3,5) : 3행5열의 2차원배열로 생성.  배열 갯수가 맞아야 함.
  arr.dtype : 배열 요소의 자료형
  arr.shape :배열 구조 행열값
  arr.ndim  : 배열의 차수
  arr.itemsize : 요소의 바이트 크기
  arr.size : 요소의 갯수
  np.zeros((행,열)) : 요소의 값이 0인 배열 생성
  np.ones((행,열)) : 요소의 값이 1인 배열 생성
                 np.ones(10,dtype=int)
  np.eye(10,10) #10행10열 단위 행렬
  np.linspace(시작값,종료값,갯수) : 시작값부터 종료값까지 갯수만큼 균등분할하는 수치
  np.pi : 원주율 상수

난수 관련 함수
   np.random.random() : 난수 발생
   np.random.default_rng(1) : seed 값 설정
   np.random.randint: 정수형 난수 리턴. 
   np.random.normal(평균,표준편차,데이터갯수) : 정규 분포 난수 생성
   np.random.choice(값의범위,선택갯수,재선택여부)
   np.random.choice(값의범위,선택갯수,확률)

통계 관련 함수
   sum,min,max,mean,std
   max(axis=1) : 행중 최대값
   max(axis=0) : 열중 최대값
   cumsum(axis=1) : 행의 누적 합계
   cumsum(axis=0) : 열의 누적 합계
   argmax(axis=1) : 행 중 최대값의 인덱스
   argmax(axis=0) : 열 중 최대값의 인덱스
   argmin(axis=1) : 행 중 최소값의 인덱스
   argmin(axis=0) : 열 중 최소값의 인덱스
   
 np.fromfunction() : 함수를 이용하여 요소의 값 설정
 arr.flat:배열의 요소들만 리턴
 np.floor: 작은 근사정수
 np.ceil : 큰 근사정수
 
 arr.ravel() #1차원배열로 변경
 arr.resize() : 배열 객체 자체를 변경


2개의 배열을 합하기
   np.vstack((i,j)) #행기준 합. 열의 갯수가 같아야 함
   np.hstack((i,j)) #열기준 합. 행의 갯수가 같아야 함.

배열 나누기
   np.hsplit(k,3) #3개로 열을 분리. 
   np.vsplit(k,2) #2개로 행을 분리. 
'''

'''
  0. 인구구조의 그래프 제목에 코드값 제거하기
'''
import numpy as np
import csv
import re 
f=open("data/age.csv")
data = csv.reader(f) #csv 형태의 파일을 읽어 저장
type(data)    #csv형태 파일
data  #반복문을 통해 한행씩 조회가능. 
import matplotlib.pyplot as plt
name="역삼"
for row in data : #data를 반복문으로 읽으면, 다시 처음부터 시작 안함.
    if row[0].find(name) >= 0 : #행정구역의 내용에 name값존재?
        print(row)
#        name=row[0]
# \( : 그룹의미하는것이 아니고 ( 문자.
# \\d* : 숫자0개이상
# \) : ) 문자
# re.sub(패턴문자,변경문자,대상문자열)
        name = (re.sub('\(\\d*\)', '', row[0]))
        #숫자의 ,제거
        row = list(map((lambda x:x.replace(",","")),row))
        print(row)
        #0세 컬럼 이후의 셀들을 배열 생성
        home = np.array(row[3:],dtype=int)
        print(home)
        break  #반복문 종료
    
#home : 해당동의 나이별 인구수를 배열로 저장
plt.style.use('ggplot') #스타일 설정
plt.figure(figsize=(10,5),dpi=100)    
plt.rc('font',family='Malgun Gothic') #한글 설정
plt.title(name+' 지역의 인구 구조')
plt.plot(home) #선그래프 출력

#같은 이름을 포함한 동이 있는 경우 모든 동을 하나의 그래프로 작성하기
f=open("data/age.csv") #f : IOStream
data=csv.reader(f) #data : age.csv 파일의 정보 저장
next(data) #1줄 읽기.
data=list(data) #파일스트림데이터를 리스트객체로 변경
                #파일내용을 리스트로 저장
data                
name="역삼"
homelist=[] #신사이름을 가진 행정동의 인구데이터 목록
namelist=[] #동이름 목록
for row in data :
    #row : 동별 인구데이터 한개.
     if row[0].find(name) >= 0 : 
         #숫자의 ,를 제거
         row = list(map((lambda x:x.replace(",","")),row))
         #row[3:] : 0세이후 인구목록
         homelist.append(np.array(row[3:], dtype = int))
         # 동의이름의 ( 이전 부분만 이름 목록에 추가
         # row[0]=서울특별시 종로구 (1111000000)  
         #row[0].find('(') : row[0]문자열에서 '('문자의 인덱스 
         namelist.append(row[0][:row[0].find('(')])
print("안녕하세요".find('하')) #하의 인덱스

row[0].find('(')
plt.style.use('ggplot')
plt.figure(figsize=(10,5),dpi=100)    
plt.rc('font',family='Malgun Gothic')
plt.title(name+' 지역의 인구 구조')
for h,n in zip(homelist,namelist) :
    #h:그래프로 출력할 데이터
    #n:동의 이름
    plt.plot(h,label=n) #하나의 그래프에 여러개의 선그래프 작성
plt.legend() #범례 출력.

#age.csv 파일을 이용하여 선택한 지역의 인구구조와 가장 비슷한
# 인구구조를 가진 지역의 그래프와 지역 출력하기
# 가장 비슷한 지역 한개만 그래프로 출력
#data : 리스트 객체
name='역삼1동' #동명을 정확하게 
for row in data:
    if name in row[0] : #row[0] 문자열에 name 포함?
        # , 를 제거. : 숫자내부의 ,제거
        row = list(map((lambda x:x.replace(",","")),row))
        #np.array(row[3:],dtype=int) : 0세이후 데이터를 배열 생성
        #                              요소는 int 형으로
        # int(row[2]) : 정수형 총인구수 
        # home : 0세이후 인구수를 정수형배열 / 총인구수 
        #        총인구수 대비 각각의 나이의 비율목록
        #        name 에 해당하는 동의 나이별 인구 비율 목록
        home = np.array(row[3:],dtype=int) / int(row[2])
        home_name = re.sub("\(\\d*\)","",row[0]) 
mn =1 
for row in data:
    # 숫자의 , 제거
    row = list(map((lambda x:x.replace(",","")),row))
    # 현재 레코드의 인구 비율 정보
    away = np.array(row[3:], dtype =int) /int(row[2])
    # name의 동과 다른지역의 데이터의 차의 제곱의 합.
    # s값이 가장 작은 지역이 name 동과 가장 비슷한 인구구조 지역
    s = np.sum((home - away) **2)
    # s < mn : 다른지역의 오차합이 더 작은지역
    #name not in row[0] : name의 지역이 아님
    if s < mn and name not in row[0] :
        mn = s
        #result_name : 현재까지 가장 오차가 적은 지역의 이름
        result_name = re.sub('\(\\d*\)', '', row[0])
        #result : 현재까지 가장 오차가 적은 지역 데이터
        result = away
        
#home : name 동의 데이터,
#home_name : name 동의 행정구역 이름
#result :data 데이터 중 가장 오차합이 작은 지역의 데이터
#result_name : 오차합이 작은 지역의 이름

plt.style.use('ggplot')
plt.figure(figsize = (10,5), dpi=100)            
plt.rc('font', family ='Malgun Gothic')
plt.title(home_name +' 지역과 가장 비슷한 인구 구조를 가진 지역')
plt.plot(home, label = home_name)
plt.plot(result, label = result_name)
plt.legend()
plt.show()

#pandas를 이용한 분석
import pandas as pd
'''
  age.csv 파일 cp949 형태의 파일임. ANSI형태. EUC-KR, KSC5601..
      csv모듈, open함수를 이용한 경우 기본인코딩이 cp949임.
      MAC은 기본인코딩방식이 UTF-8임.
   pandas 모듈에서는 기본 인코딩방식이 UTF-8임. 

 thousands="," : 숫자에 세자리마다 , 를 제거함. 숫자만읽기,
                 ,제거하고 숫자로 읽기
 index_col=0 : 0번컬럼을 인덱스로 설정.
               행정구역 컬럼이 인덱스로 설정됨                
'''
df = pd.read_csv\
  ("data/age.csv",encoding="cp949",thousands=",",
   index_col=0)
df.head()
df.info()
df.columns
#컬럼명을 변경
col_name=['총인구수','연령구간인구수']
for i in range(0,101) : #0~100
    col_name.append(str(i)+'세')
col_name    
df.columns = col_name
df.columns
#df의 모든 컬럼들을 총인구수로 나누기. 비율로 저장하기
df = df.div(df["총인구수"],axis=0)
df.head()
#총인구수,연력구간인구수 컬럼 제거
del df["총인구수"],df["연령구간인구수"]
df.info()
df.count() #컬럼의 결측값이 아닌 데이터의 건수
#결측값을 0으로 치환
#fillna : 결측값을 다른값으로 치환
df.fillna(0,inplace=True) #결측값을 0으로 치환
# 지정한 지역과 가장 비슷한 인구구조를 갖는 지역 찾아
# 그래프로 출력하기
name = "역삼1동"
# df.index : 행정구역명
# df.index.str : 인덱스의 이름을 문자열로 변경
# df.index.str.contains() : 선택된 이름을 포함?
#                    지정한 이름을 가진 레코드만 True
a = df.index.str.contains(name)
a
df2=df[a]
df2 #지정 지역 데이터
names = list(df2.index)
names[0] = names[0][:names[0].find('(')]
df2.index = names

# df3 : a값을 제외한 다른 데이터만 저장
b = list(map(lambda x : not x,a))
df3 = df[b]
mn=1
for label,content in df3.T.items() :
    #label : 행정동명
    #content : 행정동 데이터
    #s : 지정된 지역과 현재데이터의 오차 합
    s=sum((content - df2.iloc[0]) ** 2)
    if s < mn :
        mn = s;
        result = content
        name = result.name
        result.name = name[:name.find('(')]
df2.T.plot() #지정된 데이터 
result.plot()
plt.legend()        

############################################
# 데이터 전처리 : 원본데이터를 원하는 형태로 변경하는 과정
import seaborn as sns
df = sns.load_dataset("titanic")
df.info()
df.deck.unique()
#deck 컬럼의 값별 건수 출력하기
df.deck.value_counts() #결측값 제외한 값의 건수
#결측값을 포함한 값의 건수
df.deck.value_counts(dropna=False)
df.deck.head()
#isnull() : 결측값? 결측값인 경우 True, 일반값:False
df.deck.head().isnull()
#bnotnull() : 결측값아님? 결측값인 경우 False, 일반값:True
df.deck.head().notnull()

#결측값의 갯수 조회
df.isnull().sum() #컬럼별 결측값 갯수
df.isnull().sum(axis=0) #컬럼별 결측값 갯수
df.isnull().sum(axis=1) #행별 결측값 갯수
#결측값이 아닌 갯수 조회
df.notnull().sum()
df.notnull().sum(axis=0)
df.notnull().sum(axis=1)
########################
#dropna : 결측값 제거 
#         inplace=True 있어야 자체 변경 가능
# 결측값이 500개 이상인 컬럼 제거하기
# thresh=500 : 결측값의 갯수가 500 이상
df_tresh = df.dropna(axis=1,thresh=500)
df_tresh.info()
df.info()

#결측값을 가진 행을 제거
#subset=["age"] : 컬럼 설정.
#how='any'/'all' : 한개만결측값/모든값이 결측값
# axis=0 : 행
df_age = df.dropna(subset=["age"],how='any',axis=0)
df_age.info()
########################
# fillna : 결측값을 다른값으로 치환.
#          inplace=True가 있어야 자체 객체 변경
# fillna(치환할값,옵션)
#1. age 컬럼의 값이 결측값인 경우 평균 나이로 변경하기
#1. age 컬럼의 평균나이 조회하기
age_mean = df["age"].mean()
age_mean
age_mean = df.mean()["age"]
age_mean
#치환하기
df["age"].fillna(age_mean,inplace=True)
df.info()

#2. embark_town 컬럼의 결측값은 빈도수가 가장 많은 
#   데이터로 치환하기
# embark_town 중 가장 건수가 많은 값을 조회하기
#value_counts() 함수 결과의 첫번째 인덱스값.
most_freq = df["embark_town"].value_counts().index[0]
most_freq
#value_counts() 함수 결과의 가장 큰값의 인덱스값
most_freq = df["embark_town"].value_counts().idxmax()
most_freq
#  embark_town 컬럼의 결측값에 most_freq 값을 치환하기
#결측값의 인덱스 조회
df[df["embark_town"].isnull()]
df.iloc[[61,829]]["embark_town"] #결측값 확인
df["embark_town"].fillna(most_freq,inplace=True) #결측값을 수정
df.iloc[[61,829]]["embark_town"] #결측값 변경 확인
df.info()
# embarked 컬럼을 앞의 값으로 치환하기
#1.embarked 컬럼의 값이 결측값인 레코드 조회하기
df[df["embarked"].isnull()]
df.iloc[[61,829]]["embarked"]
df["embarked"][58:65] #61:C
df["embarked"][825:831] #829:Q
#앞의 데이터로 치환하기
# method="ffill" : 앞의 데이터로 치환
# method="bfill" : 뒤의 데이터로 치환
# method="backfill" : 뒤의 데이터로 치환
df["embarked"].fillna(method="ffill",inplace=True)
df["embarked"][58:65] #61:C
df["embarked"][825:831] #829:Q

#중복데이터 처리
df = pd.DataFrame({"c1":['a','a','b','a','b'],
                   "c2":[1,1,1,2,2],
                   "c3":[1,1,2,2,2]})
df
#duplicated() : 중복데이터 찾기. 
#            중복된 경우 중복된 두번째 데이터 부터 True리턴  
#            전체 행을 기준으로 중복 검색
df_dup = df.duplicated()
df_dup
df[df_dup] #중복데이터만 조회

#c1컬럼을 기준으로 중복 검색
col_dup = df["c1"].duplicated()
df[col_dup] #중복데이터 조회

#중복데 데이터를 제거하기
#drop_duplicates() : 중복된 행을 제거하기
df
#df 데이터의 중복없는 데이터 생성하기
#df 객체가 변경 안됨
df2 = df.drop_duplicates()
df2
#c1,c3 컬럼을 기준으로 중복 검색
col_dup = df[["c1","c3"]].duplicated()
df[col_dup]
#c1,c3 컬럼을 기준으로 중복 제거하기
df3 = df.drop_duplicates(subset=["c1","c3"])
df3
#auto_mpg.csv 파일 읽기
mpg = pd.read_csv("data/auto-mpg.csv")
mpg.info()
#컬럼 추가하기
#kpl : kilometer per liter mpg * 0.425
mpg["kpl"]=mpg["mpg"]*0.425
mpg.info()
mpg.kpl.head()
#kpl 컬럼의 데이터를 소숫점 1자리로 변경하기.
# 반올림하기
# round(1) : 소숫점 한자리로 반올림
mpg["kpl"] = mpg["kpl"].round(1)
mpg.kpl.head()
mpg.info()
#horsepower 컬럼의 값을 조회하기
mpg.horsepower.unique()

#오류데이터 : ? => 처리.
#             horsepower컬럼은 숫자형
# replace 함수 : 값을 변경
# ? => 결측값 변경
# np.nan : 결측값
mpg["horsepower"].replace("?",np.nan,\
                          inplace=True)
mpg.info()    
#horsepower값이 결측값인 행 조회하기
mpg[mpg["horsepower"].isnull()]
#horsepower값이 결측값인 행 삭제하기
mpg.dropna(subset=["horsepower"],axis=0,inplace=True)
mpg.info()
#자료형을 실수형 변환하기
#astype(자료형) : 모든 요소들은 자료형으로 변환.
mpg["horsepower"] = mpg["horsepower"].astype("float")
mpg.info()
