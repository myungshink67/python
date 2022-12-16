# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 09:12:38 2022

@author: KITCOOP
20221216.py
"""
'''
  지도학습 : 기계학습시 정답 제시
      회귀분석 : 예측. 회귀선을 이용하여 분석.
        독립변수(설명변수) : 
        종속변수(예측변수) : 정답
        알고리즘 : LinearRegression
        
      분류 : 알고리즘 : KNN (k-nearset-neighbors)   
                       최근접이웃알고리즘
  비지도학습 : 기계학습시 정답 제시 안함
'''
# titanic 데이터 로드
import seaborn as sns
import pandas as pd
df = sns.load_dataset("titanic")
df.info()
#전처리
#deck 컬럼 제거 : 결측값이 너무 많다.
del df["deck"]
df["embarked"].unique()
#embarked 컬럼의 결측값을 최빈값으로 변경하기
#최빈값구하기
df["embarked"].value_counts().idxmax()
most_freq = df["embarked"].value_counts().idxmax()
most_freq
df["embarked"]=df["embarked"].fillna(most_freq)
df.info()
df.head()
#설명변수 선택 
df[["class","pclass"]] #pclass 선택 
df[["embarked","embark_town"]] #embarked 선택

ndf = df[["survived","pclass","sex","age","sibsp","parch","embarked"]]
ndf.info()
#age 결측값 존재.=> 결측값이 있는 행을 제거
ndf = ndf.dropna(subset=['age'],axis=0)
ndf.info()
'''
  원핫인코딩 : 문자열형 범주데이터를 모형이 인식하도록 숫자형 변형
  pd.get_dummies()
'''
#sex 원핫인코딩하기
onehot_sex = pd.get_dummies(ndf["sex"])
onehot_sex
#ndf,onehot_sex 데이터 합하기
ndf = pd.concat([ndf,onehot_sex],axis=1)
ndf.info()
#sex 컬럼 제거
del ndf["sex"]
#embarked 원핫인코딩하기
onehot_embarked = \
    pd.get_dummies(ndf["embarked"],prefix="town")
onehot_embarked
#ndf,onehot_embarked 데이터 합하기
ndf = pd.concat([ndf,onehot_embarked],axis=1)
#embarked 컬럼 삭제하기
del ndf["embarked"]
ndf.info()
# 설명변수,목표변수 결정
# 설명변수 : survived 컬럼 제외한 변수들
# 목표변수 : survived 컬럼
X=ndf[ndf.columns.difference(["survived"])]
Y=ndf["survived"]
X.info()
X.head()
'''
   설명변수의 정규화 필요
     - 분석시 사용되는 변수의 값의 크기에 따라 영향을 미침
       age 컬럼 : 0~ 100 사이
     - 정규화 과정을 통해 설명 변수의 값을 기준 단위로 변경  
'''
from sklearn import preprocessing
import numpy as np
X = preprocessing.StandardScaler().fit(X).transform(X) 
X[5]
X.shape
#훈련데이터,테스트 데이터 분리
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = \
    train_test_split(X,Y,test_size=0.3,random_state=10)
x_train.shape    
x_test.shape
# KNN 분류 관련 알고리즘을 이용
from sklearn.neighbors import KNeighborsClassifier
#n_neighbors=5 : 5개의 최근접이웃 설정.
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train,y_train)  #학습하기
y_hat = knn.predict(x_test) #예측하기
y_hat[:20]  #예측데이터
y_test.values[:20] #실제데이터
#모형의 성능 평가하기
from sklearn import metrics
knn_report = \
    metrics.classification_report(y_test,y_hat)
knn_report
'''
support : 전체 갯수
accuracy(정확도) : 정확한 예측/전체데이터
precision(정밀도) : 실제생존자인원/생존자로 예측한인원
recall(재현율,민감도) : 생존자로 예측/실제생존한인원
f1-score(조화평균): 정밀도와 재현울 값을 이용한 성능평가지수
      2*(정밀도*재현율)/(정밀도+재현율)
 macro avg : 평균의평균
 weighted avg : 가중치 평균
 
 실제생존자 : 0 0 0 0 1 1 1 1 1 1
 예측생존자 : 0 0 0 0 1 1 1 1 1 0
   정확도 :   9/10 = 0.9
   정밀도 :   5/5  = 1.0
   재현율 :   5/6  = 0.8333
'''
#혼동행렬 : 분류 결과
knn_matrix = metrics.confusion_matrix(y_test, y_hat)
knn_matrix
'''
        예측 0   1
실제 0   [109,  16]
    1    [ 25,  65]
    
 TN : 109 : 실제 0, 예측 0
 FP :  16 : 실제 0, 예측 1
 FN :  25 : 실제 1, 예측 0
 TP :  65 : 실제 1, 예측 1
 
 실제와같은 예측 :TN,TP
 실제와다른 예측 :FN,FP
 
 정확도 : 정답/전체데이터 
         (TP+TN)/(TP+TN+FP+FN)
             174 /215 = 0.809
 정밀도 : 생존자로예측인원중 실제생존/생존자로예측인원 
         TP/TP+FP
             65 /81=0.8024691358024691
 재현율 : 실제생존자 중 생존예측/실제생존인원  
         TP/TP+FN  
             65 /90=0.7222222222222222
'''
from sklearn.metrics import accuracy_score,\
     precision_score, recall_score, f1_score
print("정확도(accuracy): %.3f" % \
                   accuracy_score(y_test, y_hat))
print("정밀도(Precision) : %.3f" % \
                   precision_score(y_test, y_hat))
print("재현율(Recall) : %.3f" % \
                   recall_score(y_test, y_hat))
print("F1-score : %.3f" % \
                   f1_score(y_test, y_hat))
    
### SVM(Support Vector Machine) 분류 알고리즘으로 
#  모델 구현하기
# SVM : 공간을(선/면)으로 분류하는 방식    
from sklearn import svm
#kernel='rbf'(기본값) 공간분리방식
#        linear, poly
svm_model = svm.SVC(kernel='rbf')
svm_model.fit(x_train, y_train) #학습
y_hat = svm_model.predict(x_test) #예측
y_hat[:10]
y_test.values[:10]
svm_report = metrics.classification_report(y_test, y_hat)  
svm_report
svm_matrix = metrics.confusion_matrix(y_test, y_hat)  
svm_matrix
print("정확도(accuracy): %.3f" % \
                   accuracy_score(y_test, y_hat))
print("정밀도(Precision) : %.3f" % \
                   precision_score(y_test, y_hat))
print("재현율(Recall) : %.3f" % \
                   recall_score(y_test, y_hat))
print("F1-score : %.3f" % \
                   f1_score(y_test, y_hat))

####
# Decision Tree (의사결정나무)
# UCI 데이터 : 암세포 진단 데이터 
# https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data
    
from sklearn import metrics
from sklearn import tree
from sklearn import preprocessing

uci_path="https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data"
#header=None : 컬럼이 없다.
df=pd.read_csv(uci_path,header=None)
df.info()
df.head()
'''
컬럼설명
1.id : ID번호 
2.clump : 덩어리 두께
3.cell_size : 암세포 크기
4.cell_shape:세포모양
5.adhesion : 한계
6.epithlial: 상피세포 크기
7.bare_nuclei : 베어핵
8.chromatin : 염색질 
9.normal_nucleoli : 정상세포
10.mitoses : 유사분열
11.class : 2 (양성), 4(악성)
'''
#df데이터에 컬럼명 설정하기
df.columns=["id","clump","cell_size","cell_shape",\
            "adhesion","epithlial","bare_nuclei",\
            "chromatin","normal_nucleoli","mitoses",
            "class"]
df.info()
#양성,악성 데이터 건수 조회하기
df["class"].value_counts()
#bare_nuclei 데이터 조회하기
df["bare_nuclei"].unique()
#bare_nuclei  ? 데이터 조회하기
df[df["bare_nuclei"]=='?'][["id","bare_nuclei","class"]]
# ?포함한 행 삭제하기
# bare_nuclei 컬럼의 자료형 정수형으로 변경하기
#? 를 결측값으로 변경
df["bare_nuclei"].replace("?",np.nan,inplace=True)
df.info()
# 결측값 행 제거
df.dropna(subset=["bare_nuclei"],axis=0,inplace=True)
df.info()
#자료형 정수로변경
df["bare_nuclei"] = df["bare_nuclei"].astype(int)
df.info()
#설명변수 : id,class 컬럼을 제외한 모든 컬럼
X=df[df.columns.difference(["id","class"])]
Y=df["class"] #목표변수
X.info()

#설명변수 정규화.
X=preprocessing.StandardScaler().fit(X).transform(X)
X[0]
#훈련/테스트 데이터 분리
x_train,x_test,y_train,y_test = train_test_split\
    (X,Y,test_size=0.3,random_state=10)
x_train.shape
x_test.shape    

#알고리즘 선택
from sklearn import tree
tree_model = tree.DecisionTreeClassifier\
    (criterion="entropy",max_depth=5)
tree_model.fit(x_train,y_train)    
y_hat=tree_model.predict(x_test)    
y_hat[:10]
y_test.values[:10]
#혼동행렬,평가레포트,
#정확도,정밀도(오류),재현율(오류),f1_score(오류)조회하기
#혼동행렬
tree_matrix=metrics.confusion_matrix(y_test, y_hat)
print(tree_matrix)
#평가레포트
tree_report=metrics.classification_report(y_test,y_hat)
tree_report
#정확도함수
accuracy_score(y_test,y_hat)  #0.9707317073170731
precision_score(y_test,y_hat) #오류 0,1인 경우만 처리가능
recall_score(y_test,y_hat)    #오류 0,1인 경우만 처리가능
f1_score(y_test,y_hat)        #오류 0,1인 경우만 처리가능
#############################################
#정밀도,재현율,f1_score 함수로 출력하기
#2->0,4->1

from sklearn import metrics
from sklearn import tree
from sklearn import preprocessing

uci_path="https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data"
#header=None : 컬럼이 없다.
df=pd.read_csv(uci_path,header=None)
df.info()
df.head()
'''
컬럼설명
1.id : ID번호 
2.clump : 덩어리 두께
3.cell_size : 암세포 크기
4.cell_shape:세포모양
5.adhesion : 한계
6.epithlial: 상피세포 크기
7.bare_nuclei : 베어핵
8.chromatin : 염색질 
9.normal_nucleoli : 정상세포
10.mitoses : 유사분열
11.class : 2 (양성), 4(악성)
'''
#df데이터에 컬럼명 설정하기
df.columns=["id","clump","cell_size","cell_shape",\
            "adhesion","epithlial","bare_nuclei",\
            "chromatin","normal_nucleoli","mitoses",
            "class"]
df.info()
#양성,악성 데이터 건수 조회하기
df["class"].value_counts()
#bare_nuclei 데이터 조회하기
df["bare_nuclei"].unique()
#bare_nuclei  ? 데이터 조회하기
df[df["bare_nuclei"]=='?'][["id","bare_nuclei","class"]]
# ?포함한 행 삭제하기
# bare_nuclei 컬럼의 자료형 정수형으로 변경하기
#? 를 결측값으로 변경
df["bare_nuclei"].replace("?",np.nan,inplace=True)
df.info()
# 결측값 행 제거
df.dropna(subset=["bare_nuclei"],axis=0,inplace=True)
df.info()
#자료형 정수로변경
df["bare_nuclei"] = df["bare_nuclei"].astype(int)
df.info()
#설명변수 : id,class 컬럼을 제외한 모든 컬럼
X=df[df.columns.difference(["id","class"])]
Y=df["class"] #목표변수
X.info()
Y.unique()
Y.replace(2,0,inplace=True) #양성 : 0
Y.replace(4,1,inplace=True) #악성 : 1
#설명변수 정규화.
X=preprocessing.StandardScaler().fit(X).transform(X)
X[0]
#훈련/테스트 데이터 분리
x_train,x_test,y_train,y_test = train_test_split\
    (X,Y,test_size=0.3,random_state=10)
x_train.shape
x_test.shape    

#알고리즘 선택
#criterion="entropy" : 불순도 설정. 
#               순수도에 도달할때 까지 반복
#     entropy : 엔트로피값
#     gini    : 지니계수
#     log_loss : 분산의 감소량을 최대화
# max_depth=5 : 트리의 깊이. 기본값 : None
#        트리의 깊이가 깊어지면 과대적합이 발생 가능 
from sklearn import tree
tree_model = tree.DecisionTreeClassifier\
    (criterion="entropy",max_depth=5)
tree_model.fit(x_train,y_train)    
y_hat=tree_model.predict(x_test)    
y_hat[:10]
y_test.values[:10]
#혼동행렬,평가레포트,
#정확도,정밀도(오류),재현율(오류),f1_score(오류)조회하기
#혼동행렬
tree_matrix=metrics.confusion_matrix(y_test, y_hat)
print(tree_matrix)
#평가레포트
tree_report=metrics.classification_report(y_test,y_hat)
tree_report
#정확도함수
accuracy_score(y_test,y_hat)  #0.9707317073170731
precision_score(y_test,y_hat) #오류 0,1인 경우만 처리가능
recall_score(y_test,y_hat)    #오류 0,1인 경우만 처리가능
f1_score(y_test,y_hat)        #오류 0,1인 경우만 처리가능

############################
# 투수들의 연봉 예측하기
############################
import pandas as pd
#1. 파일 읽기
picher = pd.read_csv("data/picher_stats_2017.csv")
picher.info()
picher.팀명.unique()
#2. 팀명을 one-hot 인코딩하기. picher데이터셋에 추가하기
onehot_team = pd.get_dummies(picher["팀명"])
onehot_team.head()
picher = pd.concat([picher,onehot_team],axis=1)
picher.info()
#3. 팀명 컬럼 제거하기
del picher["팀명"]
#4.연봉(2018) => y 컬럼으로 변경하기
picher = picher.rename(columns={"연봉(2018)":"y"})
picher.info()
#5. 독립변수,종속변수 나누기
# 독립변수 : 선수명,y컬럼을 제외한 모든 컬럼
# 종속변수 : y컬럼
X=picher[picher.columns.difference(["선수명","y"])]
Y=picher["y"]
#6. 선택된 컬럼만 정규화하기
# 표준값 : 값-평균/표준편차
def standard_scaling(df,scale_columns) :
  for col in scale_columns :
    s_mean = df[col].mean()
    s_std = df[col].std()
    df[col]=df[col].apply(lambda x:(x-s_mean)/s_std)
  return df  
#7. onehot_team.columns 은 정규화에서 제외함
scale_columns = X.columns.difference(onehot_team.columns)
scale_columns
picher.head()
picher_df=standard_scaling(X, scale_columns)
picher_df.head()
#훈련데이터와 테스트데이터로 분리하기 
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split\
    (X,Y,test_size=0.2,random_state=19)
x_train.shape    
x_test.shape
import statsmodels.api as sm
x_train = sm.add_constant(x_train) 
x_train
'''
   OLS : 선형회귀분석을 위한 모델.
         독립변수와 종속변수의 영향력을 수치로 표시
'''
model = sm.OLS(y_train,x_train).fit() #모델생성,학습
model.summary()
'''
R-squared: 결정계수. 0~1사이의값.
           독립변수의 갯수가 많아지면 값이 커짐.
Adj. R-squared : 수정 결정계수    
        표본의크기와 독립변수의 갯수 고려하여 수정
     => 독립변수의 변동량에 따른 종속변수의 변동량  
P>|t|     : p-value값
         0.05미만인경우 회귀분석에서 유의미한 피처들임
         WAR,연봉(2017),한화 3개의 피처들이 유의미한 피처
coef : 회귀계수. 독립변수별로 종속변수에미치는 영향값 수치
'''
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [20, 16]
plt.rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus']=False
coefs = model.params.tolist() #회귀계수목록
coefs
coefs_series = pd.Series(coefs)
x_labels = model.params.index.tolist()
x_labels #컬럼명
ax = coefs_series.plot(kind='bar')
ax.set_title('feature_coef_graph')
ax.set_xlabel('x_features')
ax.set_ylabel('coef')
ax.set_xticklabels(x_labels)
X.shape[1]
'''
VIF(variance_inflation_factor) : 분산팽창요인
   독립변수들을 서로 독립적이어야함.독립변수간의 연관성은
   없는게 좋다.
   다중공선성 : 독립변수들사이의 연관성으로 가중치 발생함
   예제에서는 FIP,kFIP 변수는 한개만선택하는것이 좋다
'''
from statsmodels.stats.outliers_influence \
    import variance_inflation_factor
vif = pd.DataFrame()
vif["VIF Factor"]=\
    [variance_inflation_factor(X.values,i)\
     for i in range(X.shape[1])]
vif["features"] = X.columns 
print(vif.round(1)) #소숫점 한자리로 출력

#회귀분석
from sklearn import linear_model
from sklearn import preprocessing
lr = linear_model.LinearRegression()
#독립변수 선택
X=picher_df[["FIP","WAR","볼넷/9","삼진/9","연봉(2017)"]]
Y
#훈련데이터(0.8),테스트데이터(0.2) 분리
x_train,x_test,y_train,y_test = \
  train_test_split(X,Y,test_size=0.2,random_state=19)
lr=lr.fit(x_train,y_train) #학습
predict_2018_salary = lr.predict(X) #예측.
predict_2018_salary[:5]   
Y.values[:5]

#2017연봉과 2018연봉이 다른 선수들만 10명을 
# 작년연봉,예측연봉,실제연봉 그래프 출력
#2018년 연봉이 가장 많은 선수 10명만 그래프로 작성
#예측연봉 컬럼생성
picher_df["예측연봉"]=pd.Series(predict_2018_salary)
picher_df["예측연봉"]

#2017년도 연봉
picher_df["연봉(2017)"]
picher["연봉(2017)"]
# picher_df["연봉(2017)"] 컬럼 제거
del picher_df["연봉(2017)"]
#picher["연봉(2017)"]컬럼을 picher_df["연봉(2017)"]
# 저장하기
picher_df["연봉(2017)"] = picher["연봉(2017)"]
picher_df["연봉(2017)"]
#2018년도 연봉
picher_df["y"] = picher["y"]
#선수명 저장
picher_df["선수명"] = picher["선수명"]
picher_df.info()
#2018연봉의 내림차순으로 정렬하기
result_df=\
    picher_df.sort_values(by=["y"],ascending=False)
result_df.head()    
result_df = \
    result_df[["선수명","연봉(2017)","y","예측연봉"]]
result_df.head()    
# y 컬럼을 실제연봉 변경하기
result_df = \
    result_df.rename(columns={"y":"실제연봉"})
result_df.info()    
#작년연봉과 실제연봉이 다른 선수들만 출력하기
result_df = \
result_df[result_df["연봉(2017)"] != result_df["실제연봉"]]\
    [:10]
result_df    
result_df.plot(kind="bar",x="선수명",\
   y=["연봉(2017)","실제연봉","예측연봉"])

#결정계수 : 1에 가까울수록 성능이 좋다
lr.score(x_train,y_train) #0.9150591192570362
lr.score(x_test,y_test)   #0.9038759653889865

#2. rmse 값
# mse(mean squared error) : 평균제곱오차. 작은값일수록 성능 좋다
# rmse : mse의 제곱근.    
from math import sqrt #제곱근
from sklearn.metrics import mean_squared_error #mse함수
y_pred = lr.predict(x_train) #훈련데이터 예측
sqrt(mean_squared_error(y_train, y_pred)) #7893.462873347693

y_pred2 = lr.predict(x_test) #테스트데이터 예측
sqrt(mean_squared_error(y_test, y_pred2)) #13141.866063591086

#과대적합(과적합) : 훈련데이터의 검증값 > 테스트데이터의 검증값
#                  훈련데이터의 검증값 = 테스트데이터의 검증값