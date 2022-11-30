# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 16:03:00 2022

@author: KITCOOP
test1129_a.py
"""


'''
1. main이 실행 되도록  Rect 클래스 구현하기
    가로,세로를 멤버변수로.
    넓이(area),둘레(length)를 구하는 멤버 함수를 가진다
    클래스의 객체를 print 시 :  (가로,세로),넓이:xxx,둘레:xxx가 출력
[결과]
(10,20), 넓이:200,둘레:60
(10,10), 넓이:100,둘레:40
200 면적이 더 큰 사각형 입니다.
'''    
class Rect :
    w=0  #가로
    h=0  #세로
    def __init__(self,w,h) :  #생성자
        self.w = w
        self.h = h
    def __repr__(self) :  #객체 print시 호출되는 함수
        return "(%d,%d), 넓이:%d,둘레:%d" % \
       (self.w,self.h,self.area(),self.length())
    def __gt__(self,other) :
        return self.area() > other.area()
    def __lt__(self,other) :
        return self.area() < other.area()
    def __eq__(self,other) :
        return self.area() == other.area()
    def area(self) :    #넓이
        return self.w * self.h
    def length(self) :  #둘레
        return (self.w + self.h) * 2
    
    

if __name__ == "__main__" :
     rect1 = Rect(10,20) # __init__(self,w,h)
     rect2 = Rect(10,10)
     print(rect1) # __repr__
     print(rect2)
     if rect1 > rect2 : # __gt__
         print(rect1.area(),"면적이 더 큰 사각형 입니다.")
     elif  rect1 < rect2 :   #__lt__  
         print(rect2.area(),"더 큰 사각형 입니다.")
     elif rect1 == rect2 :   #__eq__
         print(rect1.area(),"=",rect2.area(),"같은 크기의 사각형 입니다.")

'''
2. main 이 실행 되도록, Calculator 클래스를 상속받은 
   UpgradeCalculator  클래스 구현하기
   
   Calculator  클래스
     value 멤버변수
     add 멤버함수 => 현재 value의 값에 매개변수로 받는 값을 더하기
   UpgradeCalculator 클래스
     minus 멤버함수 => 현재 value의 값에 매개변수로 받는 값을 빼기
  
'''   
class Calculator:
      value=0
      #self : 멤버함수에서 매개변수 첫번째 사용
      #       자기참조변수
      def add(self, val):
          self.value += val

class UpgradeCalculator(Calculator) :
      def minus(self, val):
          self.value -= val
 

cal = UpgradeCalculator()  #객체화 
cal.add(10)
cal.minus(7)

print(cal.value) # 10에서 7을 뺀 3을 출력

'''
3. 2번에서 구현한 Calculator 클래스를 이용하여 
   MaxLimitCalculator 클래스 구현하기
MaxLimitCalculator 클래스에서 value 값은 절대 100 이상의 값을 가질수 없다.
'''
class MaxLimitCalculator(Calculator) :
      def add(self, val):   #오버라이딩 
          self.value += val
          if self.value > 100 :
              self.value = 100
    
cal = MaxLimitCalculator()
cal.add(50) # 50 더하기
print(cal.value) # 50 출력
cal.add(60) # 60 더하기
print(cal.value) # 100 출력


'''
4. 다음 코드는 알파벳 대문자의 모스 부호를 저장한 딕셔너리 데이터입니다. 
대문자 알파벳을 입력받아 알파벳의 해당하는 모스 부호를 출력하는 코드를 작성하시오 

[결과]
모스 부호로 표시할 단어(알파벳 대문자) : ABC
A : .-
B : -....
C : -.-.
'''
#code : dictionary
# value = dictionary객체[key] 
code = {'A':'.-', 'B':'-....', 'C':'-.-.', 'D':'-..', 'E':'.', 'F':'..-.', 'G':'--.',
'H':'....', 'I':'..', 'J':'.---', 'K':'-.-', 'L':'.-..', 'M':'--', 'N':'-.',
'O':'---', 'P':'.--.', 'Q':'--.-', 'R':'.-.', 'S':'...', 'T':'-', 'U':'..-',
'V':'...-', 'W':'.--', 'X':'-..-', 'Y':'-.--', 'Z':'--..'}

data1 = input("모스 부호로 표시할 단어(알파벳 대문자) :")
for i in range(len(data1)) :
    print(data1[i],":",code[data1[i]])


'''
5. 학생들의 시험 성적가 다음과 같은 경우 성적의 합계를 출력하는 코드를 작성하시오
[결과]
총합: 355 ,평균: 71.0
'''

import re #정규식 모듈 
data= 'hong:90,lee:80,kim:75,park:50,song:60'
#\d : 숫자데이터
# + : 1개이상
#\d+ : 1개이상의 숫자
pattern = re.compile("\\d+")  #패턴 생성.
#list1 : 숫자데이터들의 목록
list1 = re.findall(pattern, data)
list1 = list(map(int,list1)) #list1의 요소를 정수형으로 변경 
print("총합:",sum(list1),",평균:",sum(list1)/len(list1))


