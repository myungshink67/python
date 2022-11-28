# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 08:58:21 2022

@author: KITCOOP
20221128.py
"""

'''
  Collection : 데이터의 모임. 
    리스트(list) : 배열. 순서유지. 첨자(인덱스)사용 가능. []
    튜플(tuple) : 상수화된 리스트. 변경불가 리스트.       ()
    딕셔너리(dictionary) : (key,value)쌍 인 객체         {} 
                 items() : (key,value)쌍 인 객체를 리스트 형태로 리턴
                 keys()  : key들만 리스트형태로 리턴
                 values() : value들만 리스트형태로 리턴
    셋(set)  : 중복불가. 순서모름. 첨자(인덱스)사용 불가. 집합표현 객체.  {}
                 &, intersection() : 교집합
                 |, union() : 합집합.
                 
  컴프리헨션(comprehension) : 패턴(규칙)이 있는 데이터를 생성하는 방법
'''

###########################
#  함수와 람다
#  함수:def 예약어 사용
###########################
def func1() :
    print("func1() 함수 호출됨")
    return 10  #함수 종료 후 값을 리턴

def func2(num) :
    print("func2() 함수 호출됨:",num)
     #리턴값이 없는 함수

a=func1()
print(a)
b=func2(100)
print(b)
func2('abc')

#전역변수 : 모든 함수에서 접근이 가능한 변수
#지역변수 : 함수 내부에서만 접근이 가능한 변수

def func3() :
    c=300 #지역변수
    print("func3() 함수 호출:",a,b,c)
def func4() :
    a=110 #지역변수
    b=220 #지역변수
    print("func4() 함수 호출:",a,b)
#함수 내부에서 전역 변수값 수정하기
def func5() :
    global a,b   #a,b 변수는 전역변수를 사용함
    a=110
    b=220
    print("func5() 함수 호출:",a,b)

a=100
b=200

func3()
#print("main:",a,b,c)      #c 변수는 func3() 함수에서만 사용가능
print("main:",a,b)
func4()
print("main:",a,b)
func5()
print("main:",a,b)

#매개변수
def add1(v1,v2):
    return v1+v2
def sub1(v1,v2):
    return v1-v2

hap = add1(10,20)
sub=sub1(10,20)
print(hap)
print(sub)

hap = add1(10.5,20.1)
sub=sub1(10.5,20.1)
print(hap)
print(sub)

hap = add1("python","3.9")
print(hap)

#hap = add1("python","3.9","7")  #add1 함수의 매개변수 갯수 틀림


#가변 매개 변수 : 매개변수의 갯수를 정하지 않음 경우
def multiparam(* p) :
    result = 0
    for i in p :
        result += i
    return result
        
print(multiparam())
print(multiparam(10))
print(multiparam(10,20))
print(multiparam(10,20,30))
print(multiparam(1.5,2.5,3))
print(multiparam("1.5","2.5","3")) #매개변수 가능. 실행 오류 

#매개변수에 기본값 설정
def hap1(num1=0,num2=1) :  #매개변수가 없는 경우 0,1 기본값 설정됨
    return num1+num2

print(hap1())    #num1=0,num2=1 기본값 설정
print(hap1(10))  #num1=10,num2=1 기본값 설정
print(hap1(10,20)) #num1=10,num2=20
print(hap1(0,20))  #num1=0,num2=20
print(hap1(10,20,30)) #오류

#리턴값 두개인 경우 : 리스트로 리턴
def multiReturn(v1,v2) :
    list1=[]
    list1.append(v1+v2)
    list1.append(v1-v2)
    return list1

list1=multiReturn(200,100)
print(list1)

# 람다식을 이용한 함수
hap2=lambda num1,num2:num1+num2
print(hap2(10))  #오류 
print(hap2(10,20))  #30
print(hap2(10.5,20.5)) #31.0
#기본값 매개변수
hap3=lambda num1=0,num2=1:num1+num2
print(hap3(10))  #11
print(hap3(10,20))  #30
print(hap3(10.5,20.5)) #31.0

#문제:리스트의 평균을 구해주는 함수 getMean 구현하기
def getMean(l) :
    return sum(l)/len(l) if len(l)>0 else 0

list1 = [1,2,3,4,5,6]
print(getMean(list1))
print(getMean([]))

getMean2 = lambda l : sum(l)/len(l) if len(l)>0 else 0
print(getMean2(list1))
print(getMean2([]))

mylist1=[1,2,3,4,5]

#mylist1 보다 각각의 요소가 10이 더많은 요소를 가진 mylist2 생성
mylist2=[] #11,12,13,14,15
#1 반복문
for n in mylist1 :
    mylist2.append(n+10)
print(mylist2)    

#2 컴프리헨션
mylist2=[n+10 for n in mylist1]
print(mylist2) 
   
#3 map 방식
#map(함수,리스트) : 리스트의 각 요소에 함수 적용
def add10(n) :
    return n+10

mylist2 = list(map(add10,mylist1))
print(mylist2)    

#4 map 람다방식 
mylist2 = list(map(lambda n:n+10,mylist1))
print(mylist2)    

###### 예외 처리:예측가능한 오류 발생시 정상처리
# try except 문장

idx = "파이썬".index("일") #오류 발생
idx = "파이썬".find("일") #-1
idx

#예외처리하기
try :
    idx = "파이썬".index("이") #예외발생
    print(idx)
except :
    print("파이썬글자에는 '일'자가 없습니다.")
    
#mystr 문자열에 파이썬 문자의 위치를 strpos 리스트에 저장하기
mystr = "파이썬 공부 중입니다. 파이썬을 열심히 공부합시다"
#1
strpos = []
index=0
while True :
    index = mystr.find("파이썬",index) #index 이후부터 검색 
    if index < 0: #없다
        break
    strpos.append(index)  #0,13
    index += 1            #14
print(strpos)    
#2 index 함수 사용. 예외처리
strpos = []
index=0
while True :
  try :
    index = mystr.index("파이썬",index)
    strpos.append(index)  #0,13
    index += 1            #14
  except :  #오류 발생시 호출되는 영역
    break  
print(strpos)    

#다중예외처리 : 하나의 try 구문에 여러개의 except 구문이 존재
#              예외별로 다른 처리 가능
num1 = input("숫자형 데이터1 입력:")
num2 = input("숫자형 데이터2 입력:")
try :
    n1 = int(num1)
    n2 = int(num2)
    print(n1+n2)
    print(n1/n2)
except ValueError as e:
    print("숫자로 변환 불가")
    print(e)
except ZeroDivisionError as e :
    print("두번째 숫자는 0안됨")
    print(e)
finally :  #정상,예외 모두 실행되는 구문
    print("프로그램 종료")    
    
# 다중 예외를 하나로 묶기
num1 = input("숫자형 데이터1 입력:")
num2 = input("숫자형 데이터2 입력:")
try :
    print(a[0]) #TypeError 발생 a의 자료형 정수형
    n1 = int(num1)
    n2 = int(num2)
    print(n1+n2)
    print(n1/n2)
except (ValueError,ZeroDivisionError) as e:
    print("입력 오류")
    print(e)
finally :  #정상,예외 모두 실행되는 구문
    print("프로그램 종료")    
    
#나이를 입력받아 19세미만이면 미성년, 19세이상이면 성인 출력하기
#입력된 데이터가 숫자가 아니면 숫자만 입력하세요 메세지 출력하기
try :    #try 블럭에서 오류 발생시 except블럭으로 이동
   age = int(input("나이를 입력하세요:"))
   if age < 19 :
     print(age,":미성년")
   elif age >= 19 :    
     print(age,":성인")
except  :
   print("숫자만 입력하세요")     

#else 블럭 : 오류 발생이 안된경우 실행되는 블럭
try :    #try 블럭에서 오류 발생시 except블럭으로 이동
   age = int(input("나이를 입력하세요:"))
except  :
   print("숫자만 입력하세요")     
else : #정상적인 경우 실행 블럭
   if age < 19 :
     print(age,":미성년")
   elif age >= 19 :    
     print(age,":성인")
    
# raise : 예외 강제 발생

try :
  print(1)    
  raise ValueError
  print(2)
except ValueError :
  print("ValueError 강제 발생")   
   
#pass 예약어 : 블럭 내부에 실행될 문장이 없는 경우
n=9
if n>10 :
    pass
else :
   print("n의 값은 10 이하입니다.")    
    
try :
    age = int(input("나이를 입력하세요"))
    if age < 19:
        print("미성년")
    else :
        print("성년")  
except ValueError :
    pass    #오류 발생시 무시.   

def dumy() :
    pass

################
# 클래스 : 사용자 정의 자료형
#          구조체+함수=> 변수+함수의 모임.
# 상속 : 다중 상속 가능. 여러개의 부모클래스가 존재.
# self : 자기참조변수. 인스턴스 함수의 매개변수로 설정해야함.
# 생성자 : def __init__(self)
################
class Car :     #기본생성자 제공 클래스 : 생성자를 구현하지 않음
    color=""
    speed=0
    def upSpeed(self,value):
        self.speed += value
    def downSpeed(self,value):
        self.speed -= value        

car1 = Car()   #객체화
car1.color = "빨강"        
car1.speed = 10

car2 = Car() #객체화
car2.color = "파랑"        
car2.speed = 20
car2.upSpeed(30)
print("자동차1의 색상:%s, 현재 속도:%dkm" % (car1.color,car1.speed))
print("자동차2의 색상:%s, 현재 속도:%dkm" % (car2.color,car2.speed))

### 생성자 구현하기
class Car :
    color=""
    speed=0
    def __init__(self,v1,v2=0) :   #생성자
        self.color = v1
        self.speed = v2
        
    def upSpeed(self,value):
        self.speed += value
    def downSpeed(self,value):
        self.speed -= value        
    
car1 = Car("빨강",10) #객체화. 생성자 호출
car2 = Car("파랑",20)

car3 = Car("노랑")

car2.upSpeed(30)
print("자동차1의 색상:%s, 현재 속도:%dkm" % (car1.color,car1.speed))#빨강,10
print("자동차2의 색상:%s, 현재 속도:%dkm" % (car2.color,car2.speed))#파랑,50
print("자동차3의 색상:%s, 현재 속도:%dkm" % (car3.color,car3.speed))#노랑,0

#멤버변수 : 클래스 내부에 설정
# 인스턴스변수 : 객체별로 할당된 변수. self.변수명
# 클래스변수 : 객체에 공통변수         클래스명.변수명 

class Car :
    color="" #색상
    speed=0  #속도
    num=0    #자동차번호
    count=0  #자동차객체 갯수
    def __init__(self,v1="",v2=0) :  #생성자 
        self.color=v1  #인스턴스변수
        self.speed=v2  #인스턴스변수
        Car.count += 1 #클래스변수. 
        self.num = Car.count #인스턴스변수
    def printMessage(self) :
        print("색상:%s,속도:%dkm,번호:%d, 생산번호:%d" % \
              (self.color,self.speed,self.num,Car.count ))

car1=Car("빨강",10)
car1.printMessage() #색상:빨강,속도:10km,번호:1, 생산번호:1
car2=Car("파랑")
car1.printMessage() #색상:빨강,속도:10km,번호:1, 생산번호:2
car2.printMessage() #색상:파랑,속도:0km,번호:2, 생산번호:2

# 문제 : Card 클래스 구현하기
#   멤버변수 : kind(카드종류), number(카드숫자),
#             no(카드번호), count(현재까지 생성된 카드 갯수)
#   멤버함수 : printCard()  kind:heart, number:1,no:1,count:1
class Card :
    kind=""
    number=0
    no=0
    count=0
    def __init__(self, v1="Spade",v2=1) :
        self.kind = v1
        self.number = v2
        Card.count += 1
        self.no = Card.count
    def printCard(self) :
        print("kind:%s,number:%d,no:%d,count:%d" % \
              (self.kind,self.number,self.no,Card.count))

card1 = Card()
card1.printCard() #kind:Spade,number:1,no:1,count:1
card2 = Card("Heart")
card2.printCard() #kind:Heart,number:1,no:2,count:2
card3 = Card("Spade",10)
card1.printCard() #kind:Spade,number:1,no:1,count:3
card2.printCard() #kind:Heart,number:1,no:2,count:3
card3.printCard() #kind:Spade,number:10,no:3,count:3

#상속 : 기존의 클래스를 이용하여 새로운 클래스 생성
#       다중상속이 가능
# class 클래스명 (부모클래스1,부모클래스2,...)
class Car :
    speed = 0
    door =3
    def upSpeed(self,v) :
        self.speed += v
        print("현재 속도(부모클래스):%d" % self.speed)

# class Sedan extends Car {}
class Sedan(Car) : #Car클래스를 상속. 
   pass        # 부모클래스와 동일.

class Truck(Car) :  # Car 클래스 상속
    def upSpeed(self,v) :  #오버라이딩. 
        self.speed += v
        if self.speed > 150 :
            self.speed = 150
        print("현재 속도(자손클래스):%d" % self.speed)
    
car1 = Car()
car1.upSpeed(200)
sedan1 = Sedan()
sedan1.upSpeed(200)
truck1 = Truck()
truck1.upSpeed(200)



