# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 10:37:09 2022

@author: KITCOOP
mod2.py : 모듈예제
"""
def add(a,b) :
    print("mod2")
    return a+b

def sub(a,b) :
    print("mod2")
    return a-b

#mod2.py 직접 실행하는 경우 __name__ 변수의 값이 __main__임
# mod2.py 직접 실행하는 경우만 실행하도록 설정.
# import 되는 경우는 실행되지 않도록 설정 
if __name__ == '__main__' :  #프로그램의 시작
   print("mod2.py 실행함")
   print("add(3,4)=",add(3,4))
   print("sub(4,2)=",sub(4,2))
