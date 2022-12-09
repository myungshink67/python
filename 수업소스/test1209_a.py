# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 15:53:27 2022

@author: KITCOOP

test1209_a.py
"""
import seaborn as sns
import pandas as pd
titanic = sns.load_dataset("titanic")

'''
1 seaborn 모듈의 타이타닉 승객중 10대(10~19세)인 승객만 조회하기 
  df_teenage 데이터에 저장하기
'''
df_teenage = titanic.loc[(titanic.age >= 10) & (titanic.age < 20)]
df_teenage["age"]
df_teenage.age.value_counts()


'''
 2타이타닉 승객중 10살미만의 여성 승객만 조회하기. 
  df_female_under10 데이터에 저장하기
'''
df_female_under10 = titanic[(titanic.age < 10) & (titanic.sex == 'female')]
df_female_under10[["age","sex"]]


'''
3 동행자(sibsp)의 수가 3,4,5인 승객들의 sibsp,alone컬럼 조회하기. 
   df_notalone 데이터에 저장
'''
df_notalone = titanic.loc[(titanic.sibsp==3)|(titanic.sibsp==4)|(titanic.sibsp==5)]
df_notalone[["sibsp","alone"]]

df_notalone = titanic.loc[titanic.sibsp.isin([3,4,5])]
df_notalone[["sibsp","alone"]]

'''
 4. class 컬럼 중 First,Second인 행만 조회하기 df_class12 데이터에 저장
'''

df_class12 = titanic[titanic["class"].isin(["First","Second"])]
df_class12["class"].value_counts()

df_class12 = pd.concat([titanic.groupby("class").get_group("First"),
           titanic.groupby("class").get_group("Second")],axis=0)
df_class12["class"].value_counts()



'''
seoul.csv 파일은 https://data.kma.go.kr 사이트 기후통계분석 > 기온분석 메뉴에서 2000년1월부터 2022년6월 29일까지의 
서울의 일별 데이터를 다운받은 파일이다.
seoul.csv파일의 위치는 현재폴더의 data폴더에 존재한다고 가정한다.
파일의 encoding은 cp949로 설정한다.

5. pandas를 이용하여 파일을 읽는 코드를 작성하시오
​'''
import pandas as pd
seoul=pd.read_csv('data/seoul.csv',encoding='cp949')
print(seoul.head())

'''
6. 컬럼 명을 평균기온(℃) -> 평균기온, 최저기온(℃)->최저기온, 최고기온(℃)->최고기온으로 
   컬럼명을 변경하는 코드를 작성하시오
'''
seoul.rename(columns={'평균기온(℃)':'평균기온','최저기온(℃)':'최저기온',\
                   '최고기온(℃)':'최고기온'},inplace=True)


seoul.columns=["날짜","지점","평균기온","최저기온","최고기온"]
print(seoul.columns)


'''
7. 지점 컬럼을 삭제하는 코드를 작성하기
'''
del seoul["지점"]

seoul.drop('지점',axis=1,inplace=True)
print(seoul.head())


'''
8. 2000년 이후 서울이 가장 더웠던 날과 온도를 출력하는 코드를 작성하기
[결과]

날짜 2018-08-01
평균기온 33.6
최저기온 27.8
최고기온 39.6
Name: 6787, dtype: object

'''
seoul1=seoul.sort_values(by='최고기온',ascending=False)
seoul1.iloc[0]

#seoul.최고기온.idxmax() : 최고기온의 최대값을 가지는 레코드의 인덱스 값 리턴
seoul.iloc[seoul.최고기온.idxmax()]


'''
9. 최고기온과 최저기온의 차를 저장하는 일교차 컬럼을 생성하고, 일교차가 가장 큰날짜를 출력
하는 코드를 작성하시오

[결과]
일교차가 가장 큰 날짜: 2015-04-18 ,일교차: 18.5
'''

seoul['일교차']=seoul['최고기온']-seoul['최저기온']
seoul.head()

seoul2=seoul.sort_values(by='일교차',ascending=False)
print('일교차가 가장 큰 날짜:',seoul2.iloc[0]['날짜'],',','일교차:',\
      seoul2.iloc[0]['일교차'])

seoul2 = seoul.iloc[seoul.일교차.idxmax()]
print('일교차가 가장 큰 날짜:',seoul2['날짜'],',','일교차:',\
      seoul2['일교차'])
seoul2    


'''
10. 평균기온,최저기온,최고기온의 평균값을 구하는 코드를 작성하시오

[결과]
평균기온 12.885965
최저기온 8.996880
최고기온 17.438657
dtype: float64

'''
seoul.mean()[["평균기온","최저기온","최고기온"]]


'''
11. 월별 평균 일교차를 구하는 코드를 작성하시오. 월컬럼을 생성하기
[결과]
월 
01 7.590762
02 8.370096
03 9.242082
04 9.800000
05 10.062757
06 8.877879
07 6.587558
08 6.954992
09 8.345397
10 9.507231
11 8.272540
12 7.580031
Name: 일교차, dtype: float64
'''
seoul["월"] = seoul["날짜"].str.split("-").str.get(1)
seoul.groupby('월').mean()['일교차']
seoul.groupby('월').mean()

