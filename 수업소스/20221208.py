# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 17:05:30 2022

@author: KITCOOP
20221208.py
"""

'''
기본 함수
  np.arange(15) : 0 ~ 14까지의 숫자를 1차원 배열로 생성
  arr.reshape(3,5) : 3행5열의 2차원배열로 생성.  배열 갯수가 맞아야 함.
  arr.dtype : 배열 요소의 자료형
  arr.shape :배열 구조 행열값
  arr.ndim  : 배열의 차수
  arr.itemsize : 요소의 바이트 크기
  arr.size : 요소의 갯수
  np.zeros((행,열)) : 요소의 값이 0인 배열 생성
  np.ones((행,열)) : 요소의 값이 1인 배열 생성
                 np.ones(10,dtype=자료형)
   np.eye(10,10) #10행10열 단위 행렬
  np.linspace(시작값,종료값,갯수) : 시작값부터 종료값까지 갯수만큼 균등분할하는 수치
  np.pi : 원주율 상수

난수 관련 함수
   np.random.random() : 난수 발생
   np.random.default_rng(1) : seed 값 설정
   np.random.randint: 정수형 난수 리턴. 
   np.random.normal(평균,표준편차,데이터갯수) : 정규 분포 난수 생성
   np.random.choice(값의범위,선택갯수,재선택여부)
   np.random.choice(값의범위,선택갯수,확률)

통계 관련 함수
   sum,min,max,mean,std
   max(axis=1) : 행중 최대값
   max(axis=0) : 열중 최대값
   cumsum(axis=1) : 행의 누적 합계
   cumsum(axis=0) : 열의 누적 합계
   argmax(axis=1) : 행 중 최대값의 인덱스
   argmax(axis=0) : 열 중 최대값의 인덱스
   
 np.fromfunction() : 함수를 이용하여 요소의 값 설정
 arr.flat:배열의 요소들만 리턴
 np.floor: 작은 근사정수
 np.ceil : 큰 근사정수
 arr.ravel() #1차원배열로 변경
 arr.resize() : 배열 객체 자체를 변경


2개의 배열을 합하기
   np.vstack((i,j)) #행기준 합. 열의 갯수가 같아야 함
   np.hstack((i,j)) #열기준 합. 행의 갯수가 같아야 함.

배열 나누기
   np.hsplit(k,3) #3개로 열을 분리. 
   np.vsplit(k,2) #2개로 행을 분리. 
'''

