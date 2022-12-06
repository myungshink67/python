# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 08:56:59 2022

@author: KITCOOP
20221206.py
"""
'''
  pandas 함수
    info() : 기본 정보.
    unique() : 중복없이 한개의 데이터만 조회.
    value_counts() : 데이터별 등록된 건수. 
                     건수의 내림차순으로 정렬 출력
    groupby(컬럼명) : 컬럼의 값으로 레코드를 그룹화. 그룹별 통계자료 조회 가능.
  seaborn 모듈을 이용한 그래프
    regplot : 산점도+회귀선   
'''
import seaborn as sns
import matplotlib.pyplot as plt
titanic= sns.load_dataset("titanic")
#히스토그램 작성하기
fig = plt.figure(figsize=(15, 5))   
ax1 = fig.add_subplot(1, 3, 1) #1행3열 1번째영역
ax2 = fig.add_subplot(1, 3, 2) #1행3열 2번째영역
ax3 = fig.add_subplot(1, 3, 3) #1행3열 3번째영역
'''
sns.distplot : 밀도,빈도수 함께 출력.
               kde=False 지정하면 빈도수만 출력
sns.kdeplot : 밀도를 출력 그래프               
sns.histplot : 빈도수 출력 그래프  
'''
sns.distplot(titanic['fare'], ax=ax1)
sns.kdeplot(x='fare',data=titanic, ax=ax2)
sns.histplot(x='fare',data=titanic,ax=ax3)
ax1.set_title('titanic fare - distplot')
ax2.set_title('titanic fare - kdeplot')
ax3.set_title('titanic fare - histplot')
plt.show()

# 히트맵 그래프 : 범주형 데이터의 수치를 색상과 값으로 표시 
#pivot_table : 2개의 범주형 데이터를 행열로 분리
#aggfunc='size' : 데이터 갯수
#                 mean, sum...
table = titanic.pivot_table\
    (index=['sex'], columns=['class'], aggfunc='size')
table    

titanic[["sex","class"]].value_counts()
'''
  table : 표시할 데이터 
  annot=True : 데이터값 표시 여부
  fmt='d' : 10진 정수로 표시
  linewidth=.5 : 여백, 간격 
  cbar=False : 컬러바 표시 여부
  cmap='YlGnBu' : 컬러맵, Greys
https://matplotlib.org/3.2.1/tutorials/colors/colormaps.html     
'''
sns.heatmap(table,annot=True,fmt='d',
            cmap='YlGnBu',linewidth=.5,cbar=True)
plt.show()    

### boxplot 그래프
fig = plt.figure(figsize=(15, 10))   
ax1 = fig.add_subplot(2, 2, 1) 
ax2 = fig.add_subplot(2, 2, 2) 
ax3 = fig.add_subplot(2, 2, 3) 
ax4 = fig.add_subplot(2, 2, 4) 
'''
data=titanic : 데이터변수명.
x="alive", y="age" : titanic의 컬럼명.
hue='sex' : 성별로 분리.

violinplot : 값의범주+분포도를 표시. 가로길이 넓은부분은 분포가 많은 수치의미
'''
sns.boxplot(x='alive', y='age', data=titanic, ax=ax1) 
sns.boxplot(x='alive', y='age', hue='sex', data=titanic, ax=ax2) 
sns.violinplot(x='alive', y='age', data=titanic, ax=ax3) 
sns.violinplot(x='alive', y='age', hue='sex', data=titanic,ax=ax4) 
ax2.legend(loc="upper center")
ax4.legend(loc="upper center")
plt.show()

#pairplot : 각각의 컬럼들의 산점도출력. 대각선위치는 히스토그램으로 표시.
#           값의 분포, 컬럼간의 관계.
titanic_pair = titanic[["age","pclass","fare"]]    
titanic_pair
sns.pairplot(titanic_pair)

# FacetGrid : 조건(컬럼의 값)에 따라 그리드 나누기. 
#             컬럼의 값(범주형데이터)에 따라서 여러개의
#             그래프를 출력.
g = sns.FacetGrid(data=titanic, col='who', row='survived') 
g = g.map(plt.hist, 'age') #age 컬럼의 히스토그램 출력

###############
#  지도 시각화
import folium   #pip install folium
#location=[37.55,126.98] : 지도의 중앙 GPS값
#zoom_start=12 : 지도 확대값 
seoul_map = folium.Map(location=[37.55,126.98],zoom_start=12)
seoul_map.save("seoul.html")  #html 파일 생성.

seoul_map2 = folium.Map(location=[37.55,126.98],zoom_start=12,
                        tiles="stamenwatercolor")
seoul_map2.save("seoul2.html")  #html 파일 생성.
'''
tiles : 지도 표시되는 형식 설정.
     openstreetmap : 기본값
     cartodbdark_matter
     cartodbpositron
     cartodbpositrononlylabels
     stamentonerbackground
     stamentonerlabels
     stamenterrain, Stamen Terrain
     stamenwatercolor
     stamentoner, Stamen Toner
'''
#파일을 읽어 지도에 표시하기
import pandas as pd
import folium
#index_col=0 : 첫번째 컬럼을 index로 설정
df = pd.read_excel("data/서울지역 대학교 위치.xlsx",index_col=0)
df.info()

seoul_map = folium.Map(location=[37.55,126.98],zoom_start=12)
'''
folium.Marker : 지도에 마커 표시객체.
  [lat,lng] : 위도 경도. 마커가 표시될 위치
  popup=name : 마커 클릭시 표시되는 내용
  tooltip=name : 마커ㅓ 내부에 마우스커서가 들어온 경우 표시되는 내용
  
add_to(seoul_map) : seoul_map 지도에 추가  

zip : 목록을 하나씩 연결하여 튜플객체의 리스트로 생성 
'''
df.head()
for name,lat,lng in zip(df.index,df.위도,df.경도) :
    #name:대학교명
    #lat : 위도
    #lng : 경도
   folium.Marker\
       ([lat,lng],popup=name,tooltip=name).add_to(seoul_map)
seoul_map.save("seoul3.html")

###########
#zip :

lista = ['a','b','c']
list1 = [1,2,3]
list2 = ['가','나','다']

listall = zip(lista,list1,list2)
for d in listall :
    print(d)

#원형 마커 추가하기
df = pd.read_excel("data/서울지역 대학교 위치.xlsx",index_col=0)
seoul_map = folium.Map(location=[37.55,126.98],zoom_start=12)
for name,lat,lng in zip(df.index,df.위도,df.경도) :
   folium.CircleMarker([lat,lng],
                       popup=name,
                       tooltip=name,
                       radius=10,     #반지름크기
                       color='brown', #원둘레 색상
                       fill=True,     #원내부 채움
                       fill_color='coral', #원의 내부 색상
                       fill_opacity=0.7 #원의 내부의 투명도
                       ).add_to(seoul_map)
seoul_map.save("seoul4.html")

df = pd.read_excel("data/서울지역 대학교 위치.xlsx",index_col=0)
df.info()

#마커 내부의 아이콘 설정하기
#icon=['home','flag','bookmark','star']
seoul_map = folium.Map(location=[37.55,126.98],zoom_start=12)
for name,lat,lng in zip(df.index,df.위도,df.경도) :
   folium.Marker\
       ([lat,lng],popup=name,tooltip=name,
       icon=folium.Icon(color='red',icon='flag')
       ).add_to(seoul_map)
seoul_map.save("seoul5.html")
#Library.csv 파일을 읽어서 도서관 정보를 지도에 표시하기
df = pd.read_csv("data/Library.csv")
df.info()
#도서관명
df.시설명.head()
library_map = folium.Map(location=[37.55,126.98],zoom_start=12)
for name,lat,lng in zip(df.시설명,df.위도,df.경도) :
   folium.Marker\
       ([lat,lng],popup=name,tooltip=name,
       icon=folium.Icon(color='blue',icon='bookmark')
       ).add_to(library_map)
library_map.save("library1.html")

df.시설구분.head()
df.시설구분.unique()

'''
#시설구분별로 색상 설정하기
 시설구분 컬럼의 값에 따라
  구립,국립 : green
  사립  : red
  그외 : blue
'''
df = pd.read_csv("data/Library.csv")
df.info()
library_map = folium.Map(location=[37.55,126.98],zoom_start=12)
for name,lat,lng,kbn in zip(df.시설명,df.위도,df.경도,df.시설구분) :
   if kbn == '구립도서관' or kbn == '국립도서관' :
       color = 'green'
   elif kbn=='사립도서관' :   
       color = 'red'
   else :       
       color = 'blue'
   folium.Marker\
       ([lat,lng],popup=name,tooltip=kbn,
       icon=folium.Icon(color=color,icon='bookmark')
       ).add_to(library_map)
library_map.save("library2.html")

#MarkerCluster 기능 : 
#        지도 확대 정도에 따라 마커 표시방법을 달리해줌. 그룹화기능
from folium.plugins import MarkerCluster
library_map = folium.Map(location=[37.55,126.98],zoom_start=12)
mc = MarkerCluster()  #makercluster 객체 생성
'''
DataFrame.iterrows() : 반복문에서 한개레코드의 인덱스와 레코드값을 리턴
  _ : 변수명. 반복문에서 사용되지 않으므로 상징적인 변수로 설정
      인덱스값
  row : 한개의 레코드    
'''
for _,row in df.iterrows() :
    mc.add_child(
        folium.Marker(location=[row['위도'],row['경도']],
                      popup=row['시설구분'],
                      tooltip=row['시설명']
        )
    )
library_map.add_child(mc) #클러스터를 지도에 추가
library_map.save("library3.html")

# 경기도의 인구 데이터와 위치 정보를 이용하여 인구를 지도에 표시하기
import pandas as pd
import folium
import json
df=pd.read_excel("data/경기도인구데이터.xlsx",index_col='구분')
df.info()
df.columns #컬럼명의 자료형이 정수형
#컬럼의 자료형을 문자열형으로 변경하기
df.columns = df.columns.map(str) 
df.columns #컬럼명의 자료형이 문자열형
#2. 위치 정보를 가지고 있는 경기도행정구역경계.json 파일 읽기
# 경기도행정구역경계.json 
# 파일의 내용을 읽어서 json 형식의 객체(dict 객체)로 load
# json 형식 : {"키":"값","키2":"값2",.....}
geo_data=json.load\
    (open("data/경기도행정구역경계.json",encoding="utf-8"))
type(geo_data)
# 3. 지도 표시하기
g_map = folium.Map(location=[37.5502,126.982],zoom_start=9)
year = '2017'  
#데이터와 위치값 매칭.
folium.Choropleth(geo_data=geo_data, #위치 정보를 가진 딕셔너리 객체
     data = df[year],  #표시하고자하는 데이터값
     columns = [df.index, df[year]], #지역명,데이터
     fill_color='YlOrRd',  #채워질 색상 맵. 데이터에 따라 다른 색상 설정
     fill_opacity=0.7,    #내부 투명도
     line_opacity=0.3,    #경계선 투명도
     #데이터와 색상 표시시 사용되는 범위 지정
     threshold_scale=\
         [10000,100000,200000,300000,400000,500000,600000,700000],               
     key_on='feature.properties.name', #데이터와 지역부분 연결 값 
   ).add_to(g_map)
g_map.save('gyonggi1_' + year + '.html')

df.index
df[year]
df.loc["남양주시"]
df.loc["화성시"]
df.loc["과천시"]

import folium
import pandas as pd
state_geo = "data/us-states.json"
state_unemployment = "data/US_Unemployment_Oct2012.csv"
state_data = pd.read_csv(state_unemployment)
state_data.info()
m = folium.Map(location=[48, -102], zoom_start=3)
folium.Choropleth(
    state_geo, #문자열. 파일의 위치인식
    data=state_data,  #표시할 데이터.
    columns=["State", "Unemployment"],# 지역명,데이터 컬럼
    key_on="feature.id", #데이터값, 지도의 위치 연결 컬럼
    fill_color="YlGn",  #컬러맵
    fill_opacity=0.7,
    line_opacity=0.2,
    legend_name="Unemployment Rate (%)",  #범례명
).add_to(m)
m.save('usa1.html')
