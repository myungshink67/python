# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 16:40:07 2022

@author: KITCOOP
test1208_a.py
"""

'''
1. age.csv 파일에서 해당 지역의 인구비율과 전체지역의 인구 비율을 함께  
   그래프로 작성하기
   20221208-1.png 참조   
'''
###################### 1 #######################
import numpy as np
import csv
import matplotlib.pyplot as plt
import re
f =open('data/age.csv')
data = csv.reader(f)  #기본 인코딩:cp949. 맥:기본인코딩:UTF-8
# data = csv.reader(f,encoding="cp949") #
next(data) #한줄 읽기 첫줄을 버림.
data=list(data) #리스트로 변경. IO스트림형태는 다시 읽기 못함.
name='역삼' #행정구역이름이 역삼 이름이 있는 모든 지역 조회
homelist=[]  #역삼이 있는 행정구역의 데이터들
namelist=[]  #실제 행정 구역 이름 목록
for row in data :
     if row[0].find(name) >= 0 :
        #숫자의 ,값을 제거 
        row = list(map((lambda x:x.replace(",","")),row))
        homelist.append \
            (np.array(row[3:], dtype =int) / int(row[2]) *100) #비율%
        namelist.append(re.sub('\(\\d*\)', '', row[0])) #() 제거
        alldata = np.array(row[3:], dtype =int) / int(row[2])*100

#전체 지역의 데이터 alldata에 저장
for row in data :
    row = list(map((lambda x:x.replace(",","")),row))
    away = np.array(row[3:], dtype = int) / int(row[2]) *100
    #np.isnan : 결측값?
    if np.isnan(away).any() :  #away 데이터셋에 한개라도NA값이 존재하면 
        continue    #반복문의 처음으로 
    #vstack : 배열 행을 기준으로 합하기    
    alldata = np.vstack((alldata,away))  #행을 기준으로 연결

#연령별 연령별인구수/전체인구수 평균값
alldata = alldata.mean(axis=0)  #열별 평균
plt.style.use('ggplot')
plt.figure(figsize = (10,5), dpi=100)
plt.rc('font', family ='Malgun Gothic')
for h,n in zip(homelist,namelist) :
    plt.plot(h,label=n) 
plt.plot(alldata, label="전체")
plt.xlabel("나이")    
plt.ylabel("비율(%)")    
plt.legend()
plt.show()
plt.savefig("20221208-1.png",dpi=400,bbox_inches="tight")

###################### 2 #######################
import pandas as pd
df = pd.read_csv\
   ("data/age.csv",encoding="cp949",thousands=",",index_col=0)
df.info()
#컬럼명 다시 설정
col_name=['총인구수','연령구간인구수']
for i in range(0,101) : #0~100
    col_name.append(str(i)+'세')
col_name    
df.columns = col_name
df.columns
#df의 모든 컬럼들을 총인구수로 나누기. 비율로 저장하기
df = df.div(df["총인구수"],axis=0)
del df["총인구수"],df["연령구간인구수"]
df.info()
df.fillna(0,inplace=True) #결측값을 0으로 치환
name = "역삼"
a = df.index.str.contains(name)
a
df2=df[a]
df2 #지정 지역 데이터
df2 *= 100  #비율%
#이름들의 () 제거.
names = list(df2.index)
names = list(map(lambda x: x[:x.find('(')],names))
df2.index = names
df2.index
#전체 인구수
tot_mean = df.mean() * 100  #비율%
tot_mean.name = "전체"
df2.T.plot() #지정된 데이터 
tot_mean.plot()
plt.legend()
plt.xlabel("나이")    
plt.ylabel("비율(%)")    


'''
1-1. age.csv 파일에서 해당 지역의 인구비율과 
    전체 인구 구조와 같은 지역 찾기
'''
###################### 1 #######################
import numpy as np
import csv
import matplotlib.pyplot as plt
import re
f =open('data/age.csv')
data = csv.reader(f)
next(data) 
data=list(data) 
alldata = np.array([])
for row in data :
   row = list(map((lambda x:x.replace(",","")),row))
   away = np.array(row[3:], dtype = int) / int(row[2])
   if np.isnan(away).any() :  #away 데이터셋에 한개라도NA값이 존재하면 
      continue 
   if alldata.size == 0 :
      alldata = away
   else :   
    alldata = np.vstack((alldata,away))  #행을 기준으로 연결

#연령별 인구수/전체인구수 평균값
alldata = alldata.mean(axis=0) 

mn =1 
for row in data:
    row = list(map((lambda x:x.replace(",","")),row))
    away = np.array(row[3:], dtype =int) /int(row[2])
    s = np.sum((alldata - away) **2)
    if s < mn:
        mn = s
        result = away
        result_name =  row[0][:row[0].find('(')]
        

plt.style.use('ggplot')
plt.figure(figsize = (10,5), dpi=100)            
plt.rc('font', family ='Malgun Gothic')
plt.title('전체 인구 구조와 가장 비슷한 인구 구조를 가진 지역')
plt.plot(alldata, label = "전체")
plt.plot(result, label = result_name)
plt.xlabel("나이")    
plt.ylabel("비율")    
plt.legend()
plt.show()
plt.savefig("20221208-2.png",dpi=400,bbox_inches="tight")

###################### 2 #######################
import pandas as pd
df = pd.read_csv("data/age.csv",encoding="cp949",thousands=",",index_col=0)
df.info()
col_name=['총인구수','연령구간인구수']
for i in range(0,101) : #0~100
    col_name.append(str(i)+'세')
col_name    
df.columns = col_name
df.columns
df = df.div(df["총인구수"],axis=0)

del df["총인구수"],df["연령구간인구수"]
df.info()
df.fillna(0,inplace=True) #결측값을 0으로 치환
tot_mean = df.mean()
tot_mean.name="전체"
mn=1
for label,content in df.T.items() :
    s=sum((content - tot_mean) ** 2)
    if s < mn :
        mn = s;
        result = content
        name = result.name
        result.name = name[:name.find('(')]

tot_mean.plot() #지정된 데이터 
result.plot()
plt.legend()        
plt.title('전체 인구 구조와 가장 비슷한 인구 구조를 가진 지역')
plt.xlabel("나이")    
plt.ylabel("비율")    



'''
2. supplier_data.csv 파일을 pandas를 이용하여 읽고 
 ["1/20/14","1/30/14"] 일자 데이터만 화면에 출력하기
 isin() 함수 이용.
'''

import pandas as pd

infile='data/supplier_data.csv'
df = pd.read_csv(infile)
print(df);
print(df.info());
importdate = ["1/20/14","1/30/14"]
df_inset = df.loc[df["Purchase Date"].isin(importdate),:]
df_inset = df[df["Purchase Date"].isin(importdate)]
print(df_inset)


'''
3.  supplier_data.csv 파일 데이터에서 Invoice Number가 920으로
 시작하는 레코드만 화면에 출력하기
 startswith("920") 함수 이용 : 920문자열로 시작?
''' 
infile='data/supplier_data.csv'
df = pd.read_csv(infile)
print(df["Invoice Number"].str.startswith("920"))
df_inset = df.loc[df["Invoice Number"].str.startswith("920"),:]
df_inset = df[df["Invoice Number"].str.startswith("920")]
print(df_inset)

'''
4. sales_2013.xlsx 파일 중 Purchase Date 컬럼의 값이 
"01/24/2013"과 "01/31/2013" 인 행만 sales_2013_01.xlsx 파일로 저장하기
 isin 함수 사용.
'''
import pandas as pd
infile="data/sales_2013.xlsx"
outfile = "data/sales_2013_01.xlsx"
df = pd.read_excel(infile,"january_2013")
print(df.info())
print(df.head())
df
#엑셀파일의 날짜형태의 데이터 
select_date = ['01/24/2013','01/31/2013']
#select_date = ['2013-01-24','2013-01-31']
df_value = df[df['Purchase Date'].isin(select_date)]
df_value
print(df_value.info())
writer = pd.ExcelWriter(outfile)
df_value.to_excel(writer,sheet_name="jan_13_output",index=False)
writer.save()


'''
5. seaborn 모듈의 titanic 데이터를 이용하여 class별  
   생존 인원을 출력하시오
''' 
import seaborn as sns
titanic = sns.load_dataset('titanic')
titanic.info()
titanic.groupby(['class']).survived.sum() #생존인원수
titanic.groupby(['class'])["survived"].sum() #생존인원수
table = titanic.pivot_table\
    (index=['survived'], columns=['class'], aggfunc='size')
table


