# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 16:28:52 2022

@author: KITCOOP
test1206_a.py
"""
#1.seaborn 모듈의 iris 데이터 셋을 이용하여  품종별 산점도를 출력하기
# 20221206-1.png 파일 참조
import seaborn as sns
import matplotlib.pyplot as plt
iris = sns.load_dataset('iris')
iris.info()
iris.head()
iris.species.unique()
iris.species.value_counts()
iris.corr() #상관계수 
sns.pairplot(iris)
plt.show()
plt.savefig("20221206-1.png",dpi=400,bbox_inches="tight")



#2. iris 데이터 셋을 이용하여  각 컬럼의 값을  박스그래프로 작성하기
# 20221206-2.png 파일 참조

iris = sns.load_dataset('iris')
iris.head()
fig = plt.figure(figsize=(15, 10))   
ax1 = fig.add_subplot(2, 2, 1)
ax2 = fig.add_subplot(2, 2, 2)
ax3 = fig.add_subplot(2, 2, 3)
ax4 = fig.add_subplot(2, 2, 4)
sns.boxplot(x="species",y="sepal_length", data=iris,ax=ax1)
sns.boxplot(x="species",y="sepal_width", data=iris,ax=ax2)
sns.boxplot(x="species",y="petal_length", data=iris,ax=ax3)
sns.boxplot(x="species",y="petal_width", data=iris,ax=ax4)
plt.show()
plt.savefig("20221206-2.png",dpi=400,bbox_inches="tight")


#3. tips 데이터 셋의 total_bill 별 tip  컬럼의 회귀선을 출력하기
# 20221206-3.png 파일 참조

import seaborn as sns
import matplotlib.pyplot as plt
plt.rc('font', family="Malgun Gothic") #현재 폰트 변경 설정.

tips = sns.load_dataset('tips')
tips.head()
tips.info()
tips.day.value_counts()
tips.time.value_counts()
ax = sns.regplot(x='total_bill',y='tip',data=tips ) 
ax.set_xlabel('총지불금액')  # x축 이름 설정
ax.set_ylabel('팁') # y축 이름 설정
ax.set_title('총지불금액과 팁') # 그래프 제목 설정
plt.savefig("20221206-3.png",dpi=400,bbox_inches="tight")

#4. tips 데이터에서 점심,저녁별 tip 평균 금액을 막대그래프로 작성하기
# 20221206-4.png 파일 참조

print(tips['time'])
print(tips[tips['time']=='Lunch'].mean()["tip"])
print(tips[tips['time']=='Dinner'].mean()["tip"])
tips[tips['time']=='Dinner'].mean()["tip"]
sns.barplot(x='time',y='tip',data=tips) #barplot:  time 별 tip의 평균을 그래프
plt.savefig("20221206-4.png",dpi=400,bbox_inches="tight")


#5. tips 데이터에서 점심,저녁별 건수를 막대그래프로 작성하기
# 20221206-5.png 파일 참조

data_time = tips.time.value_counts()
data_time
#인덱스값으로 내림차순 정렬
data_time.sort_index(ascending=True,inplace=True)
plt.bar(data_time.index,data_time.values)
plt.savefig("20221206-5.png",dpi=400,bbox_inches="tight")
tips.time.value_counts() #점심,저녁별 건수 출력


'''
6. 서울시 범죄율 데이터를 이용하여 살인 정보를 지도에 표시하기
  지도 : 20221206-1.html 참조
  지도표시 데이터 : skorea_municipalities_geo_simple.json
  서울시 범죄율 데이터 : crime_in_Seoul_final.csv
'''
import json
import pandas as pd
import folium
geo_path = "data/skorea_municipalities_geo_simple.json"
geo_str = json.load(open(geo_path, encoding='utf-8')) #위치정보파일
maps_korea = folium.Map(location=[37.5502, 126.982], \
                        zoom_start=11)
df = pd.read_csv("data/crime_in_Seoul_final.csv") #데이터 정보
df=df.set_index("구별")
df.head()
maps_korea.choropleth(geo_data=geo_str, data=df["살인"],
             columns=[df.index, df["살인"]],
             fill_color='YlOrRd', fill_opacity=0.5,
             line_opacity=0.3,
             key_on="feature.properties.name"
        )
maps_korea.save('20221206-1.html')


import json
import pandas as pd
import folium
geo_path = "data/skorea_municipalities_geo_simple.json"
geo_str = json.load(open(geo_path, encoding='utf-8')) #cp949 기본인코딩방식
maps_korea = folium.Map(location=[37.5502, 126.982], \
                        zoom_start=11)
df = pd.read_csv("data/crime_in_Seoul_final.csv")
maps_korea.choropleth(geo_data=geo_str, 
             data=df,   #지도에 표시할 데이터 전체
             columns=["구별","살인"],
             fill_color='YlOrRd', fill_opacity=0.5,
             line_opacity=0.3,
             key_on="feature.id"
        )
maps_korea.save('20221206-2.html')



