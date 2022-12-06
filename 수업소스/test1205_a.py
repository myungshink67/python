# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 15:55:17 2022

@author: KITCOOP
test1205.py
"""

#1. 년도별 서울의 전입과 전출 정보를 막대그래프로 작성하여
#  20221205-1.png 파일로 그래프 저장하기
# 20221205-1.png 파일 참조
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import  rc
rc('font', family="Malgun Gothic") #현재 폰트 변경 설정.
df = pd.read_excel('data/시도별 전출입 인구수.xlsx', header=0)
df.info()
#결측값을 앞의 데이터로 채우기
#fillna() : 결측값을 다른 데이터로 변경
df = df.fillna(method='ffill') 
df.info()

mask = ((df['전출지별'] == '서울특별시') & \
        (df['전입지별'] == '전국')) 
df_seoulout = df[mask] #전출지가 서울=>전국
#df_seoulout : 서울에서 다른지역으로 나간 인구수데이터
print(df_seoulout)
#전출지별 컬럼 삭제
df_seoulout = df_seoulout.drop(['전출지별'], axis=1)
print(df_seoulout)
#전입지별 컬럼을 index로 변환
df_seoulout.set_index('전입지별', inplace=True)
print(df_seoulout)
#전국 인덱스를 전출건수 인덱스로 이름 변경
df_seoulout.rename({'전국':'전출건수'}, axis=0, inplace=True)
print(df_seoulout) #서울=>전국으로 전출한 건수 정보
mask = ((df['전입지별'] == '서울특별시') & \
        (df['전출지별'] == '전국'))
df_seoulin = df[mask] #전국 => 서울 전입 데이터.
print(df_seoulin)
df_seoulin = df_seoulin.drop(['전입지별'], axis=1)
df_seoulin.info()
#전출지별 컬럼을 인덱스로 변환 
df_seoulin.set_index('전출지별', inplace=True)
#전국 인덱스이름을 전입건수로 변경 
df_seoulin.rename({'전국':'전입건수'}, axis=0, inplace=True)
print(df_seoulin)
#pd.concat : 두개의 DataFrame 객체를 한개로 생성
df_seoul = pd.concat([df_seoulout,df_seoulin])
print(df_seoul)

#전치행렬 : 행과열을 변경
df_seoul = df_seoul.T 
print(df_seoul)

#막대그래프 출력
df_seoul.plot(kind='bar', figsize=(10, 5), width=0.7,
          color=['orange', 'green'])
plt.title('서울 전입 전출 건수', size=30)
plt.ylabel('이동 인구 수', size=20)
plt.xlabel('기간', size=20)
plt.ylim(1000000, 3500000) #y축의 데이터값의 범위.
#loc='best' : 범례의 출력 위치 가장 좋은 위치 선택
plt.legend(loc='best', fontsize=15) #범례
plt.show()
plt.savefig("202201205-1.png",dpi=400,bbox_inches="tight") #그래프를 파일로 저장 


#2. 년도별 서울의 전입과 전출 정보이용하여 순수증감인원수를 
#  선그래프로 작성하여 20221205-2.png 그래프 저장하기
# 20221205-2.png 파일 참조
import pandas as pd
import matplotlib.pyplot as plt
plt.rc('font', family="Malgun Gothic")
df = pd.read_excel('data/시도별 전출입 인구수.xlsx',\
                   header=0)
df = df.fillna(method='ffill')    
mask = (((df['전출지별'] == '서울특별시') & (df['전입지별'] == '전국')) |
        ((df['전입지별'] == '서울특별시') & (df['전출지별'] == '전국')))
df_seoul = df[mask]
df_seoul
#'전출지별','전입지별' 컬럼을 삭제 
df_seoul = df_seoul.drop(['전출지별','전입지별'], axis=1)
df_seoul.index
df_seoul.index = ["전입건수",'전출건수']
print(df_seoul)
df_seoul = df_seoul.T  #전치행렬
print(df_seoul)
#증감수 컬럼 추가하기
df_seoul["증감수"] = df_seoul["전입건수"] - df_seoul["전출건수"]
print(df_seoul)

plt.rcParams['axes.unicode_minus']=False #음수표현. -
plt.style.use('ggplot') 
#plot함수: 선그래프가 기본.
df_seoul["증감수"].plot() #기본그래프:선그래프
plt.title('서울 순수 증감수', size=20)
plt.ylabel('이동 인구 수', size=20)
plt.xlabel('기간', size=20)
plt.legend(loc='best', fontsize=15)
plt.show()
plt.savefig("20221205-2.png",dpi=400,bbox_inches="tight")


#3. 남한의 전력량을(수력,화력,원자력)을 연합막대그래프로 작성하고,
#   전력증감율을 선그래프로 작성하여 20221205-3.png 그래프로 저장하기
# 20221205-3.png 파일 참조
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot') 
plt.rcParams['axes.unicode_minus']=False 
df = pd.read_excel('data/남북한발전전력량.xlsx')
df
df = df.loc[0:4] #수력,화력,원자력, 신재생 (0~4까지)
df
print(df.head())
#axis='columns' : axis=1 같은 의미
df.drop('전력량 (억㎾h)', axis='columns', inplace=True)
print(df.head())
#발전 전력별 컬럼을 인덱스로 변경
df.set_index('발전 전력별', inplace=True)
print(df.head())
df = df.T
print(df.head())
#컬럼명 변경
df = df.rename(columns={'합계':'총발전량'})
print(df.head())
#앞의 총발전량 데이터
df['총발전량 - 1년'] = df['총발전량'].shift(1)
print(df.head())
df['증감율']=((df['총발전량'] / df['총발전량 - 1년']) - 1) * 100      
print(df)
ax1 = df[['수력','화력','원자력']].plot(kind='bar', \
             figsize=(20, 10),  width=0.7, stacked=False)  
#같은그래프 영역으로 설정    
ax2 = ax1.twinx() 
#df.index : x축 값.
#df.증감율 : y축의 값
#ls='--' : 선의 종류
ax2.plot(df.index, df.증감율, ls='--', marker='o', markersize=10, 
        color='green', label='전년대비 증감율(%)')  
ax1.set_ylim(0, 5500)
ax2.set_ylim(-50, 50)
ax1.set_xlabel('연도', size=20)
ax1.set_ylabel('발전량(억 KWh)')
ax2.set_ylabel('전년 대비 증감율(%)')
plt.title('남한 전력 발전량 (1990 ~ 2016)', size=30)
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')
plt.show()
plt.savefig("20221205-3.png",dpi=400,bbox_inches="tight")

