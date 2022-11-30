# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 16:14:13 2022

@author: KITCOOP
20221129.py
"""
'''
  함수 : def 예약어로 함수 정의
         return 값 : 함수를 종료하고 값을 전달
         매개변수 : 함수를 호출할때 필요한 인자값 정의
              가변매개변수 : 매개변수의 갯수를 지정안함. 0개이상. * p 표현
              기본값설정 : (n1=0,n2=0) : 0,1,2개의 매개변수 가능 
  예외처리 : try, except, finally, else, raise
  클래스 : 멤버변수,멤버함수, 생성자.
           인스턴스변수 : self.변수명. 객체별로 할당되는 변수
           클래스변수   : 클래스명.변수명. 해당 클래스의 모든객체들의 공통변수
    self : 자기참조변수. 인스턴스함수에 첫번째 매개변수로 설정.
   생성자 : __init__(self,...) : 객체생성에 관여하는 함수
           클래스내부에 생성자가 없으면 기본생성자를 제공.        
   상속 : class 클래스명 (부모클래스명1,부모클래스명2,..) :
          다중상속가능           
          오버라이딩 : 부모클래스의 함수를 자손클래스가 재정의
'''
#클래스에서 사용되는 특별한 함수
# __xxxx__ 형태인 함수.
class Line :
    length=0
    def __init__(self,length) :
        self.length = length
    def __repr__(self) :        #객체를 문자열로 출력
        return "선길이:" + str(self.length)        
    def __add__(self,other) : # + 연산자 사용시 호출
        print("+ 연산자 사용:",end="")
        return self.length + other.length
    def __lt__(self,other) :   # < 연산자 사용시 호출
        print("< 연산자 호출 : ")
        return self.length < other.length        
    def __gt__(self,other) :   # > 연산자 사용시 호출
        print("> 연산자 호출 : ")
        return self.length > other.length        
    def __eq__(self,other) :   # == 연산자 사용시 호출
        print("== 연산자 호출 : ")
        return self.length == other.length        

line1 = Line(200)        
line2 = Line(100)
print("line1=",line1)  
print("line2=",line2)
print("두선의 합:",line1+line2)
print("두선의 합:",line1.__add__(line2))
if line1 < line2 :
    print("line2 선이  더 깁니다.")
elif line1 == line2 :    
    print("line1과 line2 선의 길이는 같습니다,")
elif line1 > line2 :    
    print("line1 선이  더 깁니다.")
'''
클래스에서 사용되는 연산자에 사용되는 특수 함수
+   __add__(self, other)
–	__sub__(self, other)
*	__mul__(self, other)
/	__truediv__(self, other)
//	__floordiv__(self, other)
%	__mod__(self, other)
**	__pow__(self, other)
&	__and__(self, other)
|	__or__(self, other)
^	__xor__(self, other)
<	__lt__(self, other)
>	__gt__(self, other)
<=	__le__(self, other)
>=	__ge__(self, other)
==	__eq__(self, other)
!=	__ne__(self, other)

생성자 : __init__(self,...) : 클래스 객체 생성시 요구되는 매개변수에 맞도록 매개변수 구현
출력   : __repr__(self) : 클래스의 객체를 출력할때 문자열로 리턴.
'''
'''
   추상함수 : 자손클래스에서 강제로 오버라이딩 해야 하는 함수
             함수의 구현부에 raise NotImplementedError
             를 기술함
'''
class Parent:
   def method(self) :  #추상함수
       raise NotImplementedError

class Child(Parent) :
   #pass
   def method(self) :  #추상함수 오버라이딩
       print("자손클래스에서 오버라이딩함")

ch = Child()
ch.method()

#모듈 : import 모듈명
#mod1.py,mod2.py 함수를 호출하기
import mod1   #mod1.py 파일의 내용 가져옴
import mod2   #mod2.py 파일의 내용 가져옴
print("mod1 모듈 add()=", mod1.add(40,30))
print("mod1 모듈 sub()=", mod1.sub(40,30))
print("mod2 모듈 add()=", mod2.add(40,30))
print("mod2 모듈 sub()=", mod2.sub(40,30))

import mod1 as m1  #mod1 모듈의 별명 설정
import mod2 as m2  #mod2 모듈의 별명 설정
print("mod1 모듈 add()=", m1.add(40,30))
print("mod1 모듈 sub()=", m1.sub(40,30))
print("mod2 모듈 add()=", m2.add(40,30))
print("mod2 모듈 sub()=", m2.sub(40,30))

#일부만 import
#import 되는 다른 모듈의 함수 이름이 같은 경우 주의가 필요
# 어느 모듈의 함수인지 판단이 필요
from mod1 import add,sub
from mod2 import add
print("mod1 모듈 add()=", add(40,30))
print("mod1 모듈 sub()=", sub(40,30))
print("mod2 모듈 add()=", add(40,30))
print("mod2 모듈 sub()=", sub(40,30))

import mod1   #mod1.py 파일의 내용 가져옴
import mod2   #mod2.py 파일의 내용 가져옴
print("mod1 의 기본(내장) 변수:",dir(mod1))
print("mod2 의 기본(내장) 변수:",dir(mod2))
print("__name__ :", __name__) #현재파일 __name__ 변수출력
print("mod1.__name__ :", mod1.__name__)#mod1의 __name__ 변수 출력
print("mod2.__name__ :", mod2.__name__)#mod2의 __name__ 변수 출력

### 문자열의 형태 변경 변경하기
data = '''
   park 800915-1234567
   kim 890125-2345678
   choi 850125-a123456
'''
#1. 정규식 없이 주민번호 뒷자리 감추기
print(data)
result = [] #['park 800915-*******','kim 890125-*******','choi 850125-a123456']

#line : choi 850125-a123456
for line in data.split("\n"):
    word_result = [] #[choi,850125-a123456]
    #word : 850125-a123456
    for word in line.split(" ") :
        if len(word) == 14 and word[:6].isdigit() and \
            word[7:].isdigit() :
                word = word[:6]+"-"+"*******"
        word_result.append(word)
    result.append(" ".join(word_result))    
print("\n".join(result))

#2. 정규식을 이용하여 주민번호 뒷자리 감추기
import re  #정규식을 위한 모듈 
'''
re.compile(정규식패턴) : 패턴 객체 생성 
pat : 패턴객체. 정규식 형태로 정의된 형태를 지정하는 객체 
(\\d{6,7})[-]\\d{7} : 형태 지정. 패턴.
     => 앞의 6또는 7 자리 숫자, -, 7자리 숫자 인 형태 패턴
() : 그룹
\d{6,7} : \d(숫자데이터) {6,7}(6,7개의 자리수) 숫자 6또는7개
[-] : - 문자
\d{7} : 숫자 7개
'''
pat = re.compile("(\\d{6,7})[-]\\d{7}")
'''
  pat.sub : 
       문자열에서 패턴에 맞는 문자열을 찾아서 지정한 문자열로 변경
  pat.sub(변경형태,문자열)
  \g<1>-*******
  \g<1> : 첫번째 그룹 
  -     : -문자열
  ******* : 문자열
  data : 변경할 데이터 
'''
print(pat.sub("\g<1>-*******",data)) 

#정규식을 이용하여 데이터 찾기
str1 = "The quick brown fox jomps over the laze dog Te Thhhhhhe,THE"
str_list = str1.split() #공백을 기준으로 분리
print(str_list)    
#Th*e : *:0개이상. 
#       T로 시작하고 e로종료하는 문자열. 사이값이 h가 0개이상인 문자열
#       The ,Te, Thhhhhhe
pattern = re.compile("Th*e") #패턴 객체 설정
count = 0
for word in str_list:
    #pattern.search(word) : word에서 pattern에 맞는 문자열 존재?
    #                       존재: 객체리턴
    #                       없으면 : None 리턴
    if pattern.search(word) :
        count += 1
print("결과1=>%s:%d" %("갯수",count)) #1        

#re.I : 대소문자 구분없이 검색
str1 = "The quick brown fox jomps over the laze dog Te Thhhhhhe THE"
str_list = str1.split() #공백을 기준으로 분리
pattern = re.compile("Th*e",re.I) #패턴 객체 설정
count = 0
for word in str_list:
    if pattern.search(word) :
        count += 1
print("결과2=>%s:%d" %("갯수",count)) #5

#결과2에 맞는 문자열 출력하기
print("결과3=>",end="")
for word in str_list :
    if pattern.search(word) : #The,the,Te,Thhhhhhe,THE
       print(word,end=",")
print()    

print("결과4=>",re.findall(pattern, str1))
#결과2에 맞는 문자열을 aaa로 치환하기
print("결과5=>",pattern.sub('aaa', str1))
#\\d : 숫자
#문제
#str2 문자열에서 온도의 평균 출력하기
str2 = "서울:25도,부산:23도,대구:27도,광주:26도,대전:25도,세종:27도"
pattern = re.compile("\\d{2}")
tlist = re.findall(pattern,str2)
print(tlist)
tlist = list(map(int,tlist))
print(tlist)
print(sum(tlist)/len(tlist))
'''
  정규식에서 사용되는 기호
  1. () : 그룹
  2. \g<n> : n번째 그룹
  3. [] : 문자
     [a] : a 문자
     [a-z] : 소문자
     [A-Za-z] : 영문자(대소문자)
     [0-9A-Za-z] : 영문자+숫자
  4. {n} :n개 갯수
     ca{2}t : a 문자가 2개
      caat : true
      ct   : false
      cat  : false
     {n,m} :n개이상 m개이하 갯수
     ca{2,5}t : a 문자가 2개이상 5개 이하
      ct   : false
      cat  : false
      caat : true
      caaaaaaaaat : false
  5. \d : 숫자. [0-9]동일
  6. ?  : 0개또는 1개.
    ca?t : a문자는 없거나 1개    
    ct : true
    cat : true
    caat : false
  7. * : 0개이상  
    ca*t : a문자는 0개 이상
    ct : true
    cat : true
    caat : true
  8. + : 1개이상  
    ca+t : a문자는 1개 이상
    ct : false
    cat : true
    caat : true
  9. \s : 공백
     \s* : 공백문자 0개이상  
     \s+ : 공백문자 1개이상  
'''
#### 파일 읽기
'''
  open("파일명",파일모드,[인코딩])
  인코딩:파일의 저장방식. 기본값:cp949형식
  파일코드
    r :읽기
    w :쓰기. 기존의 파일의 내용을 무시. 새로운 내용으로 추가
    a :쓰기. 기존의 파일의 내용에 추가
    t : text 모드. 기본값
    b : 이진모드. binary모드. 이미지,동영상....   
'''
infp = open\
("D:/20220811/python/수업소스/20221128.py","rt",encoding="UTF-8")
while True :
    instr = infp.readline()  #한줄씩 읽기
    if instr == None or instr == '' :
        break
    print(instr,end="") #화면에 출력
infp.close()    

#파일 쓰기 : 콘솔에서 내용을 입력받아 파일로 저장하기
#현재 폴더의 data.txt 파일에 저장
outfp = open("data.txt","w",encoding="UTF-8")
while True :
    outstr = input("내용입력=>")
    if outstr == '' :
        break
    outfp.writelines(outstr+"\n") #한줄씩 파일에 쓰기
outfp.close()    

#문제 
# data.txt 파일을 읽어서 화면에 출력하기
infp = open("data.txt","r",encoding="UTF-8")
while True :
    instr = infp.readline() #한줄씩 읽기
    if instr == None or instr == "" :
        break
    print(instr,end="")
infp.close()    

infp = open("data.txt","r",encoding="UTF-8")
print(infp.read())
infp.close()

infp = open("data.txt","r",encoding="UTF-8")
print(infp.readlines())
infp.close()
'''
  readline() : 한줄씩 읽기
  read()     : 버퍼의 크기만큼 한번 읽기
  readlines() : 한줄씩 한번에 읽어서 줄별로 리스트로 리턴
'''
#이미지 파일을 읽어 복사하기
#apple.gif 파일을 읽어서 apple2.gif파일로 복사하기
infp = open("apple.gif","rb")  #원본파일. 읽기위한 파일
outfp = open("apple2.gif","wb") #복사본파일. 쓰기위한 파일
while True :
    indata = infp.read() #설정된 버퍼의 크기만큼 읽기
    if not indata :  #파일의 끝. EOF(End of File)
        break
    outfp.write(indata) #복사본파일에 데이터 쓰기
infp.close()
outfp.close()  
  
# 문제:score.txt 파일을 읽어서 점수의 총점과 평균 구하기
#
#  score.txt 내용
#  홍길동,100
#  김삿갓,50
#  이몽룡,90
#  임꺽정,80

import re
infp = open("score.txt",'r',encoding="UTF-8")
data = infp.read()
print(data)
#\\d+  :  숫자1개이상
#\\d{1,3} : 숫자 1개이상3개 이하
pattern = re.compile("\\d+") #숫자1개이상
#pattern = re.compile("\\d{1,3}") #숫자1개이상 3개이상
#data에서 숫자들 찾아서 리스트 리턴
scorelist = re.findall(pattern,data)
print(scorelist)
#요소의 자료형 int형으로 변환
scorelist = list(map(int,scorelist))
print(scorelist)
print("총합:",sum(scorelist),\
      ",평균:",sum(scorelist)/len(scorelist))

import os
#현재 작업 폴더 위치 조회
print(os.getcwd())    
#작업 폴더의 위치 변경
os.chdir("c:/Users/KITCOOP")
os.chdir("D:/20220811/python/수업소스")
#파일의 정보 조회
import os.path
file = "D:/20220811/python/수업소스"
if os.path.isfile(file) :
    print(file,"은 파일입니다.")
elif os.path.isdir(file) :    
    print(file,"은 폴더입니다.")
if os.path.exists(file) :    
    print(file,"은 존재합니다.")
else :    
    print(file,"은 없습니다.")

class Test :
    __value = 0
    def getValue(self) :
        return self.__value
    
t = Test()
print(t.__value)
print(t.getValue())
