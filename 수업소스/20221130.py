# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 08:55:39 2022

@author: KITCOOP
20221130.py
"""

'''
   클래스에 사용되는 특별한 함수들 
     __repr__,__init__, __add__ .....

    추상함수 : 반드시 오버라이딩 하도록 강제화된 함수 
              raise NotImplementedError 
              
    모듈 : import 모듈명 
           import 모듈명 as 별명 
           from 모듈명 import 함수명 => 모듈명 생략됨
           if __name__ == '__main__' : 직접 실행되는 경우만 호출
    정규식 : 문자열의 형태를 지정할수있는 방법.
            import re 모듈 사용
            패턴 = re.compile(정규식패턴) : 패턴 객체 생성
            리스트 = re.findall(패턴,문자열)
            패턴.search(문자)
            패턴.sub(치환할문자,대상문자열)
    파일 : open(파일명,모드,[encoding])
          os.getcwd() : 작업폴더
          os.chdir()  : 작업폴더 변경
          os.path.isfile(file) : 파일?
          os.path.isdir(file)  : 폴더?
    
'''
