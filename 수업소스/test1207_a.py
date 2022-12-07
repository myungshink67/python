# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 14:02:13 2022

@author: KITCOOP
test1207_a.py
"""

'''
  0. 인구구조의 그래프 제목에 코드값 제거하기
'''
import numpy as np
import csv
import re 
f=open("data/age.csv")
data = csv.reader(f) #csv 형태의 파일을 읽어 저장
type(data)    
data  #반복문을 통해 한행씩 조회가능
import matplotlib.pyplot as plt
name="역삼"
for row in data :
    if row[0].find(name) >= 0 : #행정구역의 내용에 name값존재?
        print(row)
#        name=row[0]
        name = (re.sub('\(\d*\)', '', row[0]))
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

'''
1. 임의의 값으로 10*10 배열을 만들고, 전체 최소값과 최대값, 
 행별 최대값과 최소값, 열별 최대값과 최소값을 출력하기
  임의의 값이므로 결과에 표시된 숫자는 다름
[결과]
a 최대값 : 0.9957371929996585
a 최소값 : 0.013959842183549176
a 행별 최대값 : [0.85702828 0.99573719 ... 0.99251539 0.93112787]
a 행별 최소값 : [0.04483455 0.08025441 ... 0.27046588 0.05850934]
a 열별 최대값 : [0.93112787 0.91484578 ... 0.62135503 0.95952193]
a 열별 최소값 : [0.08247677 0.04483455 ... 0.05383681 0.05850934 0.09736856] 
'''
import numpy as np
rg=np.random.default_rng(2) #난수생성기 seed값. 데이터 복원시 필요 
a= rg.random((10,10)) #10행 10열 배열
#a = np.random.random((10,10)) #seed값 설정안함
a
a.shape
print("a 최대값 :",a.max())
print("a 최소값 :",a.min())
print("a 행별 최대값 :",a.max(axis=1))
print("a 행별 최대값 인덱스:",a.argmax(axis=1))
print("a 행별 최소값 :",a.min(axis=1))
print("a 행별 최소값 인덱스:",a.argmin(axis=1))
print("a 열별 최대값 :",a.max(axis=0))
print("a 열별 최대값 인덱스:",a.argmax(axis=0))
print("a 열별 최소값 :",a.min(axis=0))
print("a 열별 최소값 인덱스:",a.argmin(axis=0))



'''
2. 임의의 값을 30개 저장하는 배열을 만들고 평균값을 출력하기
  임의의 값이므로 결과에 표시된 숫자는 다름
[결과]
0.44769045640141436
'''
import numpy as np
b= np.random.random((30)) #seed값 설정 없음
b
b= np.random.random(30)
b
b.mean()



'''
3.  결과와 같은 값을 저장하고 있는 8*8 행렬을 생성하기
[결과]
[[0. 1. 0. 1. 0. 1. 0. 1.]
 [1. 0. 1. 0. 1. 0. 1. 0.]
 [0. 1. 0. 1. 0. 1. 0. 1.]
 [1. 0. 1. 0. 1. 0. 1. 0.]
 [0. 1. 0. 1. 0. 1. 0. 1.]
 [1. 0. 1. 0. 1. 0. 1. 0.]
 [0. 1. 0. 1. 0. 1. 0. 1.]
 [1. 0. 1. 0. 1. 0. 1. 0.]]

'''

c = np.zeros((8,8)) #8행8열 배열 생성. 요소값은 0으로 설정
c
c[1::2,::2] = 1
c
c[::2,1::2] = 1
print(c)


'''
4. 0부터 10까지의 요소를 가진 배열을 생성하고
그중 3에서 8사이의 모든 요소를 음수인 값을 갖는
배열을 생성
[결과]
[ 0,  1,  2,  3, -4, -5, -6, -7,  8,  9, 10]
'''

d=np.arange(11) #0부터 10까지의 요소를 가진 배열
d
d[4:8] = -d[4:8]
d

d=np.arange(11) #0부터 10까지의 요소를 가진 배열
d
d[(3<d) & (d<8)] *= -1
d


'''
5. 0부터 9까지 정수형 난수 100개를 요소로 가진 배열 중 
그중 3의 배수인 값은 음수 값을 갖는 배열을 생성
[결과]
array([-9,  5,  2,  0,  1,  2,  7,  8,  7,  0, -6, -3, -9,  0, -9, -3,  8,
        7,  8,  1, -6, -6, -9,  8,  1,  1, -3,  0, -9, -9,  4,  0,  5,  8,
        7,  7,  7,  7,  2,  2,  5, -9, -3,  1,  2,  5,  2,  8, -9,  7, -9,
        5,  0,  1, -9, -9,  7, -3,  4,  2,  0,  1,  8,  1,  7,  1,  4,  1,
       -3, -6,  8, -3, -9,  7, -9,  0, -9,  2,  1,  1,  8,  2,  2, -3,  7,
        8, -9,  7,  1, -3,  4,  5, -6, -9,  7,  0, -3,  7,  0,  2])
'''
f=np.random.randint(10,size=100)
f[(f%3==0)] *= -1 #3의 배수만 True
f
