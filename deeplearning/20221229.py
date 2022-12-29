# -*- coding: utf-8 -*-
"""
Created on Thu Dec 29 10:17:19 2022

@author: KITCOOP
20221229.py
"""
'''
   인공신경망(ANN) 
    단위 : 퍼셉트론
    
    y = x1w1 + x2w2 + b
    x1,x2 : 입력값, 입력층
    y : 결과값.
    w : 가중치
    b : 편향.
'''
import numpy as np
def AND(x1,x2) :  #1,0
    x = np.array([x1,x2])  #입력값
    w = np.array([0.5,0.5]) #가중치
    b = -0.8                #편향
    tmp = np.sum(w*x) + b
    if tmp <= 0 :
        return 0
    else :
        return 1
    
for xs in [(0,0),(0,1),(1,0),(1,1)] :
   y=AND(xs[0],xs[1])  
   print(xs,"=>",y)

#퍼셉트론 OR 게이트 구현   
def OR(x1,x2) :  #1,0
    x = np.array([x1,x2])  #입력값
    w = np.array([0.5,0.5]) #가중치
    b = -0.2                #편향
    tmp = np.sum(w*x) + b
    if tmp <= 0 :
        return 0
    else :
        return 1
    
for xs in [(0,0),(0,1),(1,0),(1,1)] :
   y=OR(xs[0],xs[1])  
   print(xs,"=>",y)   
   
#퍼셉트론 NAND 게이트 구현   
def NAND(x1,x2) :  
    x = np.array([x1,x2])  #입력값
    w = np.array([0.5,0.5]) #가중치
    b = -0.2                #편향
    tmp = np.sum(w*x) + b
    if tmp <= 0 :
        return 0
    else :
        return 1
    
for xs in [(0,0),(0,1),(1,0),(1,1)] :
   y=NAND(xs[0],xs[1])  
   print(xs,"=>",y)      