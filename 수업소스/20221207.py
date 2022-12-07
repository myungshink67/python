# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 09:35:03 2022

@author: KITCOOP

20221207.py
"""
## numpy :행렬, 통계관련기본함수,배열 기능 제공하는 모듈
import numpy as np
#배열 생성
#np.arange(15) : 0 ~ 14까지의 숫자를 1차원 배열로 생성
#reshape(3,5) : 3행5열의 2차원배열로 생성.
#               배열 갯수가 맞아야 함.
np.arange(15)
a = np.arange(15).reshape(3,5)
a  #0~14까지의 숫자를 3행 5열의 2차원배열로 생성
type(a)

#배열 요소의 자료형
a.dtype  #int32 => 32비트, 4바이트

#배열 형태
a.shape #(3,5) : 3행 5열 2차원 배열

np.arange(15).shape #(15,) 1차원배열
np.arange(15).reshape(15,1).shape #(15, 1) 2차원배열

#배열의 차수
a.ndim  #2차원
np.arange(15).ndim #1

#배열의 요소의 바이트 크기
a.itemsize  #4
#배열의 요소의 갯수
a.size 
np.arange(15).size

#리스트로 배열 생성하기
b=np.array([6,7,8])
b
type(b)
#튜플로 배열 생성하기
c=np.array(6,7,8)  #오류
c=np.array((6,7,8))
c
type(c)
#리스트로 2차원 배열 생성하기
d=np.array([[6,7,8],[1,2,3]])
d

#0으로 초기화된 3행 4열 배열 e 생성하기
e=np.array([[0,0,0,0],[0,0,0,0],[0,0,0,0]])
e
e.shape

e=np.zeros((3,4))
e
e.shape

# 모든 요소의 값이 0인 배열 100개를 1차원으로 생성하기
f = np.zeros(100)
f.shape

# 모든 요소의 값이 1인 배열 100개를 1차원으로 생성하기
g = np.ones(100)
g.shape
#1으로 초기화된 10행 10열 배열 h생성하기
h =np.ones((10,10))
h
h.shape
#0~9999까지의 값을 가진 배열을 100행 100열 배열 i생성하기
i = np.arange(10000).reshape(100,100)
i.shape
i
#0~2까지의 숫자를 9개로 균등분할하여 배열 생성
j=np.linspace(0,2,9)
j
j.size
#0~9까지의 숫자를 20개로 균등분할한 배열 생성
k=np.linspace(0,9,20)
k
k.size
#정수형 1의 값으로 10개를 가진 배열 l
l = np.ones(10,dtype=int)
l
l.dtype

#상수값 
np.pi

# numpy 데이터 연산
#1차원 배열의 연산
a = np.array([20,30,40,50])
b = np.arange(4) #(0,1,2,3)
c = a-b  #각각의 요소들을 연산
c # array([20, 29, 38, 47])

c = a+b  #각각의 요소들을 연산
c # array([20, 29, 38, 47])

c = b**2 #b 요소들 각각의 제곱
c

c = a < 35 # a배열의 요소을 35와 비교하여 작으면 True,크면 False
c

#2차원 배열의 연산
a=np.array([[1,1],[0,1]])
b=np.array([[2,0],[3,4]])
a
b
c = a + b #각각의 요소를 연산
c
c = a - b #각각의 요소를 연산
c
c = a * b #각각의 요소를 연산
c
# @: 행렬의 곱. dot
c = a @ b
c
'''
  a    @   b  =    c 
행 [1,1] [2,0]   [1*2+1*3][1*0+1*4]  [5,4]
   [0,1] [3,4]   [0*2+1*3][0*3+1*4]  [3,4]
'''
c = a.dot(b)
c

#난수를 이용한 배열 생성
rg = np.random.default_rng(1) #seed값 설정
rg
a = rg.random((2,3)) #2행3열배열. 
a
b=np.ones((2,3),dtype=int)
b
a.dtype
b.dtype
c = a+b #실수형 = 실수형+정수형
c = b+a #실수형 = 실수형+정수형
c

a+=b  #실수형=실수형+정수형
a
a.dtype

b+=a  #오류. 정수형 = 정수형+실수형

#a배열의 전체 요소들의 합
a.sum()
#a배열의 전체 요소들 중 최소값
a.min()
#a배열의 전체 요소들 중 최대값
a.max()
#a배열의 전체 요소들 중 평균값
a.mean()
#a배열의 전체 요소들 중 중간값
a.median()
#a배열의 전체 요소들 중 표준편차값
a.std()
a
#a배열의 행 중 최대값
a.max(axis=1)
#a배열의 열 중 최대값
a.max(axis=0)
#a배열의 행 중 최소값
a.min(axis=1)
#a배열의 열 중 최소값
a.min(axis=0)
#a배열의 행 별 합계값
a.sum(axis=1)
#a배열의 행 별 누적합계값
a.cumsum(axis=1) 

#a배열의 열 별 합계값
a.sum(axis=0)
#a배열의 열 별 누적합계값
a.cumsum(axis=0) 

#10부터 49까지의 c배열을 생성하기
c=np.arange(10,50)
c
#첫번째 값 출력하기
c[0]
#첫번째~4번째까지 값 출력하기
c[:4] #0번인덱스부터 ~ 3번인덱스까지
c[0:4] #0번인덱스부터 ~ 3번인덱스까지
#4번인덱스값을 100으로 변경
c[4]=100
c[:5]
#처음부터 3씩 증가하여 10인덱스까지 조회
c[:11:3]
c[:11]

#0부터11까지의 숫자를 3행4열 배열 d로 생성하기
d=np.arange(12).reshape(3,4)
d
#1행1열의 값을 조회하기
d[1,1]
d[0:2,0:2] #1행까지, 1열까지 조회
d[:2,:2] #1행까지, 1열까지 조회
d[::2,::2] #2씩증가. 
#1값으로 채워진 10행 10열 배열 e 생성하기
e=np.ones((10,10))
e
#e배열의 가장 자리는 1로 내부는 0으로 채워진 배열로수정하기
e[1:9,1:9]=0
e
e[1:-1,1:-1]=0
e

#e배열과 같은 모양의배열 f 생성하기
f = np.zeros((8,8))
f
'''
   행과 열에 각각 1인 값의 행열을 추가하기
   pad() : 행,열을 추가.
   pad_width=1 : 추가될 행과열의 갯수
   constant_values = 1 : 추가되는 값
'''
f=np.pad(f,pad_width=1,constant_values = 1)
f
f.shape

#np.fromfunction() : 함수를 이용하여 요소의 값 설정
#np.fromfunction(함수명,행열,요소자료형)
def f(x,y) : #x:행의 인덱스, y:열의인덱스
    return 10*x+y
'''
 f(0행,0열) : 0
 f(0행,1열) : 1
 f(0행,2열) : 2
 f(0행,3열) : 3
 f(1행,0열) : 10
   ..
 f(2행,0열) : 20
  
 g [0,1,2,3]
   [10,11,12,13]
   [20,21,22,23]
   ..
'''
g=np.fromfunction(f,(5,4),dtype=int)
g

#g배열의 0행 출력하기
g[0]
#g배열의 0열 출력하기
g[:,0] # :=>모든 행, 0:0열
#g배열의 2열 출력하기
g[:,2] # :=>모든 행, 0:2열
#g배열의 0,1행, 0,1열 출력하기
g[:2,:2]

#g.flat:배열의 요소들만 리턴
for i in g.flat :
    print(i)

#난수를 이용하여 0~9사이의 정수값을 가진 임의의수를 3행4열
# 배열 생성
#np.floor: 작은 근사정수
#np.ceil : 큰 근사정수
h=np.floor(np.random.random((3,4)) * 10)
h
h.ndim
h.shape
#h배열을 1차원배열 h1 변경하기
h1=h.ravel() #h배열이 변경되지 않음
h1.ndim
h1.shape
#h배열을 6행2열 배열 h2 변경하기
h2=h.reshape(6,2) #h배열이 변경되지 않음.
h2.shape
h.shape
#h배열 자체를 6행2열의 배열로 변경하기
h.resize(6,2)
h.shape
#h배열을 3행5열로 변경
h.reshape(3,5) #오류. h배열의 요소의 갯수가 12. 3행5열:15개
h.reshape(3,4)
h.shape
#3행을 지정. 열을 자동 맞춤
h.reshape(3,-1) #-1을 지정하면 자동으로 맞춤.
h.reshape(4,-1).shape #-1을 지정하면 자동으로 맞춤.
h.reshape(-1,4).shape

#0~9사이의 정수형 난수값을 가진 2행2열 배열 생성
#randint: 정수형 난수 리턴. 
i=np.random.randint(10,size=(2,2))
i
j=np.random.randint(10,size=(2,2))
j

#2개의 배열을 합하기
np.vstack((i,j)) #행기준 합. 열의 갯수가 같아야 함
np.hstack((i,j)) #열기준 합. 행의 갯수가 같아야 함.
#배열 나누기
k = np.random.randint(10,size=(2,12))
k
np.hsplit(k,3) #3개로 열을 분리. 
np.vsplit(k,2) #2개로 행을 분리. 
k.shape
#k배열의 모든 요소값을 100으로 변경하기
#k=100. k변수에 100 정수값을 저장. k값은 배열이 아님.
k.shape
k[0,0]=100
k[0,:]=100
k[:]=100
k[:,:]=200
k
#0~19사이의 임의의 정수를 가진 5행 4열 배열 l을 생성하기
l = np.random.randint(20,size=(5,4))
l
l.max()
#각 행의 최대값들을 조회하기
l.max(axis=1)
#각 열의 최대값들을 조회하기
l.max(axis=0)

#각 행의 최대값의 인덱스들 조회하기
l.argmax(axis=1)
#각 열의 최대값의 인덱스들 조회하기
l.argmax(axis=0)

#각 행의 최소값의 인덱스들 조회하기
l.argmin(axis=1)
#각 열의 최소값의 인덱스들 조회하기
l.argmin(axis=0)

#단위행렬 : 대각선(행==열) 셀의 값이 1인 배열
m=np.eye(10,10) #10행10열 단위 행렬
m
#m배열의 0이아닌 요소의 인덱스 조회하기
np.nonzero(m)
n=[1,2,0,4,0] #리스트
type(n)
np.nonzero(n) #요소의 값이 0이아닌 요소의 인덱스 리턴
