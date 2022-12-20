# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 08:59:10 2022

@author: KITCOOP
20221220.py
"""
######################################
# data/review_data.csv 읽어서 df에 저장하기
######################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import time
df=pd.read_csv("data/review_data.csv")
df.info()

#한글,공백 부분만 전달
def text_cleaning(text) :
    '''
        [^ ㄱ-ㅣ가-힣]+ : 공백,한글이 아닌경우
        ^ : not
        ㄱ-ㅣ(모음이) : 자음, 모음 한개
        가-힣  : 가 부터 힣 한글
        + : 한개 이상
    '''
    nhangul = re.compile('[^ ㄱ-ㅣ가-힣]+')
    #정규식에 맞는 데이터를 빈문자열로 치환
    result = nhangul.sub("",text)
    return result  #한글 또는 공백만 리턴

data = text_cleaning\
("!!!***가나다 123 라마사아 ㅋㅋㅋ 123 fff")
data
#리뷰에서 한글과 공백만 남김.
df["ko_text"] =df["review"].apply\
    (lambda x : text_cleaning(str(x)))
df.info()
#ko_text 컬럼의 내용이 있는 경우만 df에 다시저장하기
# strip() : 양쪽 공백 제거
df=df[df["ko_text"].str.strip().str.len() > 0]
df.info()
df.head()
df.review.head()
#review 컬럼 삭제
del df["review"]

#한글 형태소 분리
from konlpy.tag import Okt

def get_pos(x) :
    okt = Okt()
    pos = okt.pos(x)
    #컴프리헨션 방식 리스트 객체
    pos=['{0}/{1}'.format(word,t) for word,t in pos]
    return pos
result = get_pos(df["ko_text"].values[0])
result
df.info()
#글뭉치 변환하기 : 단어들을 인덱스화
from sklearn.feature_extraction.text import CountVectorizer
#글뭉치 :분석을 위한 글모임.
index_vectorizer=CountVectorizer\
               (tokenizer=lambda x : get_pos(x)) #(맛집/Noun)
#df["ko_text"].tolist() : 분석할 데이터 목록
# 문장을 형태소 분석 후 인덱스 설정
'''
   안녕 나는 홍길동 이야  
     1    2     3    4    => 인덱스화
   반가워 나는 김삿갓 이야
     5    2     6     4
     
     1 2 3 4 5 6
     (2,6)
'''
X = index_vectorizer.fit_transform(df["ko_text"].tolist())
X.shape  #(556,4498). 행의수(df의 행의수),4498(단어수)
for a in X[0] :
    print(a)
'''
TfidfTransformer 
   TF-IDF(Term Frequency-Inverse Document Frequency)
   TF : 한문장에 등장하는 단어의 빈도수
      예: 현재 문장에서 맛집 단어가 3번등장 => 3
   IDF : 전체문서에서 등장하는 단어의 빈도수의 역산
      예: 전체 문서에서 맛집 단어가 10번 등장 => 1/10 => 0.1
   TF-IDF
        3*0.1 = 0.3
   => 전체 문서에서 많이 나타나지 않고, 현재문장에서 많이 나타나는 경우
      현재 문장에서 해당 단어의 중요성을 수치로 표현     
'''
from sklearn.feature_extraction.text import TfidfTransformer
tfidf_vectorizer = TfidfTransformer()
X = tfidf_vectorizer.fit_transform(X)
X.shape #(556, 4498)
print(X[0])

y=df["y"]  #0:부정, 1:긍정
y

#훈련데이터,테스트 데이터로 분리
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test =\
                  train_test_split(X,y,test_size=0.3)
x_train.shape # (389, 4498)
x_test.shape  # (167, 4498)

#Logistic Regression 알고리즘을 이용하여 분류
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(random_state=0)
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)
y_pred[:10]  #예측데이터
y_test.values[:10] #실제 데이터
# 평가하기
#혼동행렬
from sklearn.metrics import confusion_matrix
confmat = confusion_matrix(y_test,y_pred)
confmat
#정확도,정밀도,재현율,f1-score 조회하기
from sklearn.metrics import \
    accuracy_score,precision_score,recall_score,f1_score
print("정확도:",accuracy_score(y_test, y_pred))
print("정밀도:",precision_score(y_test, y_pred))
print("재현율:",recall_score(y_test, y_pred))
print("f1_score:",f1_score(y_test, y_pred))

#각피처 별 가중치값 조회하기
lr.coef_[0]
len(lr.coef_[0])
#가중치값을 그래프로 출력하기
plt.rcParams["figure.figsize"] = [10,8]
plt.bar(range(len(lr.coef_[0])),lr.coef_[0])
# 긍정의 가중치 값 5개 : 
#    가중치계수를 내림차순 정렬하여 최상위 5개 값
sorted(((value,index) for index,value \
    in enumerate(lr.coef_[0])),reverse=True)[:5]
# 부정의 가중치 값 5개 : 
#    가중치계수를 오름차순 정렬하여 최상위 5개 값
sorted(((value,index) for index,value \
    in enumerate(lr.coef_[0])),reverse=False)[:5]
sorted(((value,index) for index,value \
    in enumerate(lr.coef_[0])),reverse=True)[-5:]
#회귀계수값으로 정렬하기
coef_pos_index = sorted(((value,index) \
     for index,value in enumerate(lr.coef_[0])),\
     reverse=True)
coef_pos_index[:5]  #긍정    
coef_pos_index[-5:] #부정   

#index_vectorizer : 단어들을 인덱스화 한 객체
#index_vectorizer.vocabulary_ : 딕셔너리객체
# (k(형태소단어),v(형태소단어의인덱스))

# invert_index_vectorizer : 딕셔너리 객체
# (k(형태소단어의인덱스),v(형태소단어))
invert_index_vectorizer = {v:k for k,v in \
  index_vectorizer.vocabulary_.items()}
     
cnt = 0
for k,v in index_vectorizer.vocabulary_.items() :
   print(k,v)
   cnt += 1
   if cnt >= 10 :
       break
   
#invert_index_vectorizer : {형태소인덱소:형태소단어값}
#  딕셔너리 객체
#상위 20개의 긍정 형태소 출력하기
#coef_pos_index : 회귀계수의 내림차순 정렬된 데이터
#invert_index_vectorizer[coef[1]] :형태소값
#coef[1] : 컬럼명. 형태소의 인덱스 
for coef in coef_pos_index[:20] :
   print(invert_index_vectorizer[coef[1]],coef[0])

#상위 20개의 부정 형태소 출력하기
for coef in coef_pos_index[-20:] :
   print(invert_index_vectorizer[coef[1]],coef[0])

# 명사(Noun) 기준으로 긍정 단어 10개, 
# 부정 단어  10개의 단어 출력하기 
noun_list=[]
for coef in coef_pos_index :
    noun = invert_index_vectorizer[coef[1]].split("/")[1]
    if noun == 'Noun' :
      noun_list.append\
          ((invert_index_vectorizer[coef[1]],coef[0]))
         
noun_list[:10] #긍정 명사
noun_list[-10:] #부정 명사
# 형용사(Adjective) 기준으로 긍정 단어 10개, 
# 부정 단어  10개의 단어 출력하기 
adj_list=[]
for coef in coef_pos_index :
    adj = invert_index_vectorizer[coef[1]].split("/")[1]
    if adj == 'Adjective' :
      adj_list.append\
          ((invert_index_vectorizer[coef[1]],coef[0]))
adj_list[:10]
adj_list[-10:]

####################################################
#  비지도 학습 : 목표변수,종속변수가 없음. 
#  군집 : 데이터를 그룹(클러스터)화.
####################################################
import pandas as pd
import matplotlib.pyplot as plt
#고객의 연간 구매금액을 상품 종류별로 구분한 데이터
uci_path = 'https://archive.ics.uci.edu/ml/machine-learning-databases/\
00292/Wholesale%20customers%20data.csv'
df = pd.read_csv(uci_path, header=0)
df.info()
df.head()
X = df.iloc[:,:]
X
#데이터 정규화
from sklearn import preprocessing
X=preprocessing.StandardScaler().fit(X).transform(X)
X    
from sklearn import cluster
#n_clusters=5 : 5개의 그룹 분리
#init="k-means++" : 중심점 선정을 위한 알고리즘 설정
#  k-means++ : 기본값. 확률분포를 기반으로 샘플링
#  random    : 무작위로 관측값 설정
#  n_init=10 : 10개 중심점으로 시작
kmeans = cluster.KMeans(init="k-means++",n_clusters=5,\
                        n_init=10)
kmeans.fit(X) #학습하기. 비지도학습이므로 y값 설정 안함
cluster_label = kmeans.labels_ #그룹화한 결과
cluster_label
len(cluster_label)  #440. 행의갯수. 0~4까지의 값
df["cluster"]=cluster_label
df.info()
df["cluster"].unique() #0~4
df["cluster"].value_counts()
#cluster별 평균값 조회하기
df.groupby("cluster").mean()
#산점도 
#Grocery ,Frozen 컬럼을 산점도 출력하기
df.plot(kind="scatter",x="Grocery",y="Frozen",
        c="cluster",cmap="Set1",colorbar=True,
        figsize=(10,10))
#Milk ,Delicassen 컬럼을 산점도 출력하기
df.plot(kind="scatter",x="Milk",y="Delicassen",
        c="cluster",cmap="Set1",colorbar=True,
        figsize=(10,10))

df.groupby("cluster").mean()["Grocery"]
#academy1.csv 파일을 읽어서 3개로 (KMeans 알고리즘)군집화하기
data=pd.read_csv("data/academy1.csv")
data
model = cluster.KMeans(init="k-means++",n_clusters=3)
#data.iloc[:,1:] : 행전체, 0번컬럼제외. 학번컬럼을 제외
model.fit(data.iloc[:,1:])
result=model.predict(data.iloc[:,1:]) #클러스터 리턴
result
data["group"]=result
plt.rc("font",family="Malgun Gothic")
data.plot(kind="scatter",x="국어점수",y="영어점수",
        c="group",cmap="Set1",colorbar=True,
        figsize=(7,7))

#group 별 평균
data.groupby("group").mean()

# iris 데이터를 이용하여 군집 알고리즘
from sklearn import datasets
iris = datasets.load_iris()
iris
type(iris)
iris.data  #설명변수. 
iris.data.shape #(150, 4)
iris.target #목표변수. 품종코드
iris.target.shape #(150,)
labels=pd.DataFrame(iris.target)
labels.info()
labels.columns=["labels"]
labels
datas=pd.DataFrame(iris.data)
datas.info()
datas.columns=["Sepal length","Sepal width",\
               "Petal length","Petal width"]
data=pd.concat([datas,labels],axis=1)    
data.info()
#꽃받침 정보로 그룹화하기
feature = data[["Sepal length","Sepal width"]]
model = cluster.KMeans(n_clusters=3)
model.fit(feature)
model.labels_
data["group"]=model.labels_
data.info()
#예측 데이터로 그래프 출력
fig=plt.figure()
plt.scatter(data["Sepal length"],\
            data["Sepal width"],
            c=data["group"],alpha=0.5)
fig=plt.figure()
data.plot(kind="scatter",x="Sepal length",y="Sepal width",
        c="group",cmap="Set1",colorbar=True,
        figsize=(7,7))
#실제 데이터로 그래프 출력
data.plot(kind="scatter",x="Sepal length",y="Sepal width",
        c="labels",cmap="Set1",colorbar=True,
        figsize=(7,7))
    
#####################################
# 수업안함
#  DBScan :  
#####################################
# 군집 : DBSCAN 알고리즘. => 공간의 밀집도로 클러스터 구분 
import pandas as pd
import folium
file_path = 'data/2016_middle_shcool_graduates_report.xlsx'
df = pd.read_excel(file_path,  header=0)
df.info()
# df데이터에서 각 중학교의 정보를 지도로 표시하기
mschool_map = folium.Map(location=[37.55,126.98], zoom_start=12)
for name,lat,lng in zip(df.학교명,df.위도,df.경도) :
    folium.CircleMarker([lat,lng],
                        radius=5,color='brown', fill=True,fill_color='coral',
                  fill_opacity=0.7, popup=name,tooltip=name).add_to(mschool_map)
mschool_map.save('seoul_mscshool.location.html')    
df.info()
df.지역.unique()
df.코드.unique()
df.유형.unique()
# 원핫인코딩 : 문자열 -숫자형. 컬럼 구분.
#            preprocessing.OneHotEncoder()
# label인코더 : 크기정보가 의미 없이, 단순한 종류인 경우.
#               문자/숫자 -> 숫자형
#            preprocessing.LabelEncoder()
from sklearn import preprocessing
label_encoder=preprocessing.LabelEncoder()
onehot_encoder=preprocessing.OneHotEncoder()
label_code = label_encoder.fit_transform(df["코드"])
label_code
df["코드"].values[:10]
label_code[:10]
df["코드"].unique()
label_loc = label_encoder.fit_transform(df["지역"])
label_loc
label_type = label_encoder.fit_transform(df["유형"])
label_type
label_day = label_encoder.fit_transform(df["주야"])
label_day
#label_encoder 된 데이터를 df의 컬럼으로 추가
df["code"]=label_code
df["location"]=label_loc
df["type"]=label_type
df["day"]=label_day
df.info()
df["code"].unique()
df["location"].unique()
df["type"].unique()
df["day"].unique()
df.info()
#속성변수 설정. 과학고,외고국제고,자사고 진학률로 분리하기
X=df.iloc[:,[9,10,13]]
X.info()
#X 데이터 정규화
X = preprocessing.StandardScaler().fit(X).transform(X)
X[:5]
# DBSCAN 알고리즘
# eps=0.2 : 반지름 크기. 
# min_samples=5 : 최소점의 갯수.
#                 eps 영역내에 최소 5개의 점이 있으면 클러스터로 인정
# cluster : 그룹
# core point : 그룹화를 위한 중심점.
# noise point : 그룹화 하지 못한 데이터. -1 설정

dbm = cluster.DBSCAN(eps=0.2,min_samples=5)
dbm.fit(X)
cluster_label = dbm.labels_
df["cluster"] = cluster_label
df["cluster"].unique()
df["cluster"].value_counts()

#cluster로 그룹화 하여 레코드 조회하기
for key,group in df.groupby("cluster") :
    print("* cluster:", key)
    print("* number :",len(group))
    print(group.iloc[:,[0,1,3,9,10,13]].head())
    print("\n")

