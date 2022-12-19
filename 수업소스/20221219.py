# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 08:37:59 2022

@author: KITCOOP
20221219.py
"""
'''
    titanic 속성
pclass : Passenger Class, 승객 등급
survived : 생존 여부
name : 승객 이름
sex : 승객 성별
age : 승객 나이
sibsp : 탑승 한 형제/배우자 수
parch : 탑승 한 부모/자녀 수
ticket : 티켓 번호
fare : 승객 지불 요금
cabin : 선실 이름
embarked : 승선항 (C = 쉘 부르그, Q = 퀸즈타운, S = 사우스 햄튼)
body : 사망자 확인 번호
home.dest : 고향/목적지
'''
import pandas as pd
df_train = pd.read_csv("data/titanic_train.csv")
df_test = pd.read_csv("data/titanic_test.csv")
df_train.info()
df_test.info()
# df_train 데이터에서 생존여부를 그래프로 출력하기
df_train["survived"].value_counts()
df_train["survived"].value_counts().plot.bar()
df_train["survived"].value_counts().plot(kind="bar")
# 좌석 등급별 생존여부 조회하기
df_train.groupby("pclass")["survived"].value_counts()
df_train[["pclass","survived"]].value_counts()
#index로 정렬
df_train.groupby("pclass")["survived"].\
    value_counts().sort_index()
df_train[["pclass","survived"]].value_counts().\
    sort_index()
#그래프 출력
df_train[["pclass","survived"]].value_counts().\
    sort_index().plot(kind="bar")
# df_train 데이터에서 pclass별 건수를 그래프    
#   hue="survived" : 건수를 survived 컬럼의 값을 분리
#   y축의값 : 건수. 
import seaborn as sns
sns.countplot(x="pclass",hue="survived",data=df_train)
df_train.info()
df_test.info()
# 1. age 컬럼의 결측값을 
# df_train데이터의 평균값으로 변경하기 (df_train,df_test)
age_mean = df_train["age"].mean()
age_mean
df_train["age"]=df_train["age"].fillna(age_mean)
df_test["age"]=df_test["age"].fillna(age_mean)
df_train.info()
df_test.info()
#2. embarked 컬럼의 결측값을 최빈값으로 변경
embarked_freq = df_train["embarked"].value_counts().idxmax()
embarked_freq
df_train["embarked"]=df_train["embarked"].fillna(embarked_freq)
df_train.info()
#3. name,ticket,cabin,body,home.dest 컬럼 제거하기
df_train = df_train.drop\
    (["name","ticket","cabin","body","home.dest"], axis=1)
df_test = df_test.drop\
    (["name","ticket","cabin","body","home.dest"], axis=1)    
#4. df_train,df_test 데이터를 통합하기.
whole_df = df_train.append(df_test)   
whole_df.info()
#훈련데이터의 갯수 => 훈련데이터/테스트 데이터 분리
train_num = len(df_train)
#원핫인코딩하기
whole_df_encoded =pd.get_dummies(whole_df)
whole_df_encoded.info()

#5. 설명변수,목표변수 분리.
df_train = whole_df_encoded[:train_num]
df_test = whole_df_encoded[train_num:]
#훈련 설명변수
x_train = df_train.loc\
      [:,df_train.columns != "survived"].values
x_train.shape      
#훈련 목표변수
y_train = df_train["survived"].values      

#테스트 설명변수
x_test = df_test.loc\
      [:,df_test.columns != "survived"].values      
x_test.shape      
#테스트 목표변수     
y_test = df_test["survived"].values      
#로지스틱 회귀분석 모델을 이용하여 분류하기
#로지스틱 회귀분석 분류 알고리즘.
#0~1사이의 값을 리턴.
# 0.5미만 : 0, 0.5이상:1
from sklearn.linear_model import LogisticRegression
# 설명변수들 사이의 가중치. 임의로 선택.
# 모델의 재현을 위한 설정. 
lr = LogisticRegression(random_state=0)
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)
y_pred[:10]
y_test[:10]
#분류모델평가
#1. 혼동행렬
from sklearn.metrics import confusion_matrix
con_mat = confusion_matrix(y_test,y_pred)
con_mat
#2. 정확도, 정밀도,재현율,f1score 출력
from sklearn.metrics import accuracy_score,\
    recall_score,precision_score,f1_score
print("정확도:",accuracy_score(y_test,y_pred))
print("정밀도:",precision_score(y_test,y_pred))
print("재현율:",recall_score(y_test,y_pred))
print("f1-score:",f1_score(y_test,y_pred))

# cabin(선실) 컬럼을  추가하여 예측하기
import pandas as pd
df_train = pd.read_csv("data/titanic_train.csv")
df_test = pd.read_csv("data/titanic_test.csv")
df_train.info()
df_test.info()
# 1. age 컬럼의 결측값을 
# df_train데이터의 평균값으로 변경하기 (df_train,df_test)
age_mean = df_train["age"].mean()
age_mean
df_train["age"]=df_train["age"].fillna(age_mean)
df_test["age"]=df_test["age"].fillna(age_mean)
df_train.info()
df_test.info()
#2. embarked 컬럼의 결측값을 최빈값으로 변경
embarked_freq = df_train["embarked"].value_counts().idxmax()
embarked_freq
df_train["embarked"]=df_train["embarked"].fillna(embarked_freq)
df_train.info()

#3. df_train,df_test를 whole_df 데이터 합하기
whole_df=df_train.append(df_test)
train_num = len(df_train)
whole_df.info()
whole_df["cabin"].unique()
'''
E36 => E
B96 B98 => B
cabin 컬럼의 첫번째 문자만 추출하여 cabin 컬럼에 저장
'''
whole_df["cabin"]=whole_df["cabin"].str[0]
whole_df["cabin"].unique()
#결측값을 X로 치환하기
whole_df["cabin"] = whole_df["cabin"].fillna("X")
whole_df["cabin"].value_counts()
#G,T데이터를 X로 치환하기
whole_df["cabin"]=whole_df["cabin"].replace({"G":"X","T":"X"})
whole_df["cabin"].value_counts()
#cabin 별 생존자별 건수 그래프
sns.countplot(x="cabin",hue="survived",data=whole_df)

#name 피처 활용
whole_df["name"].head()
#name의 ,를 기준으로 두번째 문자열이 당시의 사회적 지위에 해당 하는 문자.
#name_grade 컬럼에 저장하기
name_grade=whole_df["name"].apply\
    (lambda x:x.split(", ")[1].split(".")[0])
name_grade.unique()
name_grade.value_counts()
#호칭에 따라, 사회적 지위를 나타냄. 
#비슷한 지위로 표시
grade_dict = {
 "A":["Rev","Col","Major","Dr","Capt","Sir"], #명예직
 "B":["Ms","Mme","Mrs","Dona"],  #여성
 "C":["Jonkheer","the Countess"], #귀족
 "D":["Mr","Don"],               #남성
 "E":["Master"],                 #젊은 남성
 "F":["Miss","Mlle","Lady"]      #젊은 여성
  }

def give_grade(g) :
    #k=F, v=["Miss","Mlle","Lady"]
    for k,v in grade_dict.items() :
        for title in v:
            if g == title :
                return k
    return 'G'

#Miss=>F, Mr=>D
name_grade = \
    list(name_grade.map(lambda x : give_grade(x)))            
name_grade[:10]
whole_df["name"]=name_grade
whole_df["name"].value_counts()
#name별로 생존여부를 그래프로 출력하기
sns.countplot(x="name",hue="survived", data=whole_df)
whole_df.info()
#ticket,body,home.dest 컬럼 삭제
whole_df = whole_df.drop\
    (["ticket","body","home.dest"],axis=1)
whole_df.info()
#whole_df  데이터를 one-hot 인코딩하기    
whole_df_encoded = pd.get_dummies(whole_df)
whole_df_encoded.info()
# x_train,x_test, y_train, y_test 데이터 분리.
df_train=whole_df_encoded[:train_num]
df_test=whole_df_encoded[train_num:]
x_train = \
  df_train.loc[:,df_train.columns !='survived'].values
y_train=df_train["survived"].values
x_test = \
  df_test.loc[:,df_test.columns !='survived'].values
y_test=df_test["survived"].values

from sklearn.linear_model import LogisticRegression
# 설명변수들 사이의 가중치. 임의로 선택.
# 모델의 재현을 위한 설정. 
lr = LogisticRegression(random_state=0)
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)
y_pred[:10]
y_test[:10]
#분류모델평가
#1. 혼동행렬
from sklearn.metrics import confusion_matrix
con_mat = confusion_matrix(y_test,y_pred)
con_mat
#2. 정확도, 정밀도,재현율,f1score 출력
from sklearn.metrics import accuracy_score,\
    recall_score,precision_score,f1_score
print("정확도:",accuracy_score(y_test,y_pred))
print("정밀도:",precision_score(y_test,y_pred))
print("재현율:",recall_score(y_test,y_pred))
print("f1-score:",f1_score(y_test,y_pred))

#피처별 영향력을 그래프로 출력하기
import numpy as np
import matplotlib.pyplot as plt
cols = df_train.columns.tolist()
cols.remove("survived")
x_pos = np.arange(len(cols))
plt.rcParams["figure.figsize"]=[5,4]
fig,ax = plt.subplots()
ax.barh(x_pos,lr.coef_[0],
        align="center",color="green",ecolor="black")
ax.set_yticks(x_pos)
ax.set_yticks(x_pos)
ax.set_yticklabels(cols)
ax.invert_yaxis()
ax.set_xlabel("Coef")
ax.set_title("Each Feature's Coef")
plt.show()
'''
  한글 분석을 위한 모듈 : konlpy
  pip install konlpy
  시스템환경변수:JAVA_HOME 환경설정필요
  
  형태소 분석 모듈
  Okt(Open korean Text)
  Kkma(코코마)
  Komoran(코모란)
  Hannanum(한나눔)
'''
from konlpy.tag import Okt,Kkma,Komoran,Hannanum
import time
okt=Okt()
kkma=Kkma()
komoran = Komoran()
han=Hannanum()

def sample_ko_pos(text):
 print(f"==== {text} ====")
 start = time.time() #현재시간
 #pos(text):text를 형태소를 분리. 품사표시
 print("kkma:",kkma.pos(text),",실행시간:",time.time()-start)
 start = time.time()
 print("komoran:",komoran.pos(text),",실행시간:",time.time()-start)
 start = time.time()
 print("okt:",okt.pos(text),",실행시간:",time.time()-start)
 start = time.time()
 print("hannanum:",han.pos(text),",실행시간:",time.time()-start)
 print("\n")

text1 = "영실아 안녕 오늘 날씨 어때"
sample_ko_pos(text1)
text2 = "영실아안녕오늘날씨어때"
sample_ko_pos(text2)
text3 = "안녕 ㅎㅏㅅㅔ요 ㅈㅓ는 ㄷㅐ학생입니다."
sample_ko_pos(text3)

#카카오맵을 크롤링하여 맛집리뷰에 사용되는 용어 분석하기
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from selenium import webdriver
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
import re
import time
path = 'C:/setup/chromedriver.exe'
source_url = "https://map.kakao.com/"
driver = webdriver.Chrome(path)
driver.get(source_url) #웹브라우저에 카카오맵지도 
time.sleep(1)
# searchbox : html 중 id="search.keyword.query" 인태그
searchbox=driver.find_element(By.ID,"search.keyword.query")
searchbox.send_keys("강남역 고기집")
time.sleep(1)
#searchbutton : 검색버튼. id="search.keyword.submit"
searchbutton=driver.find_element\
    (By.ID,"search.keyword.submit") 
#arguments[0].click() :  searchbutton 클릭
driver.execute_script("arguments[0].click();", searchbutton)    
time.sleep(1)
#브라우저의 현재 화면의 html 소스를 html변수 저장.
html = driver.page_source
#html 문장을 분석
soup = BeautifulSoup(html, "html.parser")
#moreviews : class="moreview" a태그목록. 상세보기 들
moreviews = soup.find_all\
    (name="a", attrs={"class":"moreview"})
page_urls = [] #강남역 고기집의 조회된 상세보기 href 목록
for moreview in moreviews:
    page_url = moreview.get("href") #a태그의 href 속성의 값
    page_urls.append(page_url) 
driver.close()  #브라우저 종료  
print(page_urls)
print(len(page_urls))    

########################################################
#   장인닭갈비 후기 읽어서 DataFrame으로 저장
columns = ['score', 'review']
page='https://place.map.kakao.com/95713992'
df = pd.DataFrame(columns=columns) #컬럼만 존재.
df
driver = webdriver.Chrome(path) #브라우저 실행
driver.get(page) #고기집 상세화면
time.sleep(2)
another_reviews = driver.find_element(By.CSS_SELECTOR,"span.txt_more")
print(another_reviews.text)
for i in range(11): #11번을 후기 더보기 텍스트 클릭
    time.sleep(2)
    another_reviews = driver.find_element(By.CSS_SELECTOR,"span.txt_more")
    try :
       another_reviews.text.index('후기 더보기')
       another_reviews.click() #후기 더보기 클릭
    except :
        break  #후기더보기 값이 없으면 오류발생. 오류 발생 반복문 종료

time.sleep(2)
html = driver.page_source  #html 소스데이터 
soup = BeautifulSoup(html,'html.parser')
contents_div = soup.find(name='div', attrs={"class":"evaluation_review"}) #리뷰 영역. 별점,내용
reviews = contents_div.find_all(name="p",attrs={"class":"txt_comment"}) #리뷰목록
rates=contents_div.find_all(name="span",attrs={"class":"inner_star"})  #별점값 목록
for rate, review in zip(rates, reviews):  #(별점목록,리뷰목록)
    rate_style = rate.attrs['style'].split(":")[1] #width:20%
    rate_text=int(rate_style.split("%")[0])/20 #1
    row = [int(rate_text), review.find(name="span").text]
    series = pd.Series(row, index=df.columns) #score,review 인덱스 설정. 시리즈객체로 생성.
    df = df.append(series, ignore_index=True)

driver.close()
df.info()
df.head()
df.score.value_counts()

########################################

#############################################
#상세보기에 조회된 고기집 목록을 조회
columns = ['score', 'review']
df = pd.DataFrame(columns=columns) #컬럼만 존재.
driver = webdriver.Chrome(path) #브라우저 실행
page_urls
### 반복문 시작
for idx, page in enumerate(page_urls):
   print(idx+1,page)
   driver.get(page) #고기집 상세화면
   time.sleep(2)
   another_reviews = driver.find_element(By.CSS_SELECTOR,"span.txt_more")
   for i in range(20):
       time.sleep(2)
       another_reviews = driver.find_element(By.CSS_SELECTOR,"span.txt_more")
       try :
          another_reviews.text.index('후기 더보기')
          another_reviews.click() #후기 더보기 클릭
       except :
           break

   time.sleep(2)
   html = driver.page_source  #html 소스데이터 
   soup = BeautifulSoup(html,'html.parser')
   try :
      contents_div = soup.find(name='div', attrs={"class":"evaluation_review"}) #리뷰 영역. 별점,내용
      reviews = contents_div.find_all(name="p",attrs={"class":"txt_comment"}) #리뷰목록
      rates=contents_div.find_all(name="span",attrs={"class":"inner_star"})  #별점값 목록

      for rate, review in zip(rates, reviews):  #(별점목록,리뷰목록)
          rate_style = rate.attrs['style'].split(":")[1]
          rate_text=int(rate_style.split("%")[0])/20
          row = [int(rate_text), review.find(name="span").text]
          series = pd.Series(row, index=df.columns) #score,review 인덱스 설정. 시리즈객체로 생성.
          df = df.append(series, ignore_index=True)
   except :
        continue

###################### 반복문 종료 
driver.close()
df.info() 
df.head()
df["score"].value_counts()

#별점 5,4 => 긍정(1)
#별점 1,2,3 => 부정(0)
df["y"]=df["score"].apply\
    (lambda x : 1 if float(x)>3 else 0)
df.info() 
df["y"].value_counts()
#df 데이터를 review_data.csv 파일로 저장
df.to_csv("data/review_data.csv",index=False)
