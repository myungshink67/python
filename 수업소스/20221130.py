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
            리스트 = re.findall(패턴,문자열) : 
                    문자열에서 패턴에 해당하는 문자열의 목록 리턴
            패턴.search(문자)   : 패턴에 맞는 문자?
            패턴.sub(치환할문자,대상문자열) :치환

    파일 : open(파일명,모드,[encoding])
          os.getcwd() : 작업폴더 조회
          os.chdir()  : 작업폴더 변경
          os.path.isfile(file) : 파일?
          os.path.isdir(file)  : 폴더?
    
'''
#폴더의 하위 파일 목록 조회 
import os
print(os.listdir())
file="data.txt"
os.path.exists(file) #존재?
#문제 : 작업파일의 하위파일목록 출력하기
# 파일인 경우 : 파일의 크기 os.path.getsize(파일명)
# 폴더인 경우 : 하위파일의 갯수
# 작업폴더의 하위파일갯수 
len(os.listdir())
#현재 작업폴더
cwd = os.getcwd();
cwd
for f in os.listdir() :
    if os.path.isfile(f) :
        print(f,":파일, 크기:",os.path.getsize(f))
    elif os.path.isdir(f) :
        os.chdir(f)
        print(f,":폴더, 하위파일의갯수:",len(os.listdir()))
        os.chdir(cwd)
        
#폴더 생성
os.mkdir("temp") #temp 폴더 생성
#폴더 제거
os.rmdir("temp") #temp 폴더 제거

#######엑셀 파일 읽기
import openpyxl 
'''
   xlsx : openpyxl  모듈사용
   xls  : xlrd 모듈로 읽기
          xlwd 모듈로 쓰기
'''
filename = "data/sales_2015.xlsx"
#엑셀파일 전체 
book = openpyxl.load_workbook(filename)
# 첫번째 sheet
sheet = book.worksheets[0]
data=[]
for row in sheet.rows :
    line = []
    print(row)
    #enumerate(row) : 목록에서 
    #                 l : 인덱스
    #                 d : 데이터. 셀의값
    for l,d in enumerate(row) :
        line.append(d.value) #셀의내용을 line 추가
#    print(line) #한 줄의 셀의 리스트
    data.append(line)
#print(data)    

#xls 형식의 엑셀파일 읽기
import xlrd 
infile = "data/ssec1804.xls"
#workbook : 엑셀파일 전체 데이터
workbook = xlrd.open_workbook(infile)
#workbook.nsheets : sheet의 갯수
print("sheet 의 갯수",workbook.nsheets) #26
for worksheet in workbook.sheets() :
    #worksheet : 한개의 sheet 데이터
    print("worksheet 이름:",worksheet.name)
    print("행의 수:",worksheet.nrows)
    print("컬럼의 수:",worksheet.ncols)
    #worksheet.nrows : 행의 수
    #worksheet.ncols : 컬럼의 수
    for row_index in range(worksheet.nrows) : 
        for column_index in range(worksheet.ncols) :
            print\
        (worksheet.cell_value(row_index,column_index),",",end="")
        print()   
        
'''
  csv,엑셀파일 => 표(테이블,그리드)형태 데이터 
                  행,열로 이루어진 데이터
        => pandas 모듈을 이용하여 표형태의 데이터로 처리.
           DataFrame 형식으로 처리함.   
'''        

### sqlite : 파이썬 내부에 존재하는 데이터 베이스
# https://sqlitebrowser.org/dl/
# windows-64비트용 zip sqlite browser 다운받기 
# c:\ 압축풀기
import sqlite3
dbpath = "test.sqlite"  #database 파일 이름. 
conn = sqlite3.connect(dbpath) #데이터 베이스 접속.
cur = conn.cursor() # sql 구문을 실행할 수 있는 객체 
# executescript : 여러개의 sql 문장을 실행.
#                 각각의 문장들은 ;으로 구분됨      
'''
  drop table if exists items; => items 테이블이 존재하면 테이블 삭제.
  => items 테이블 생성
  item_id integer primary key : 
               item_id 컬럼이 숫자형 기본키. 값이 자동증가됨
  name text unique : 문자형 데이터. 중복불가

  create table items (item_id integer primary key,
        name text unique, price integer);
=> insert 구문 실행. 
=> item_id 컬럼을 제외 : 값이 자동 증가됨  
  insert into items (name,price) values ('Apple',800);                  
  insert into items (name,price) values ('Orange',500);
  insert into items (name,price) values ('Banana',300);  

'''
cur.executescript("""
  drop table if exists items;
  create table items (item_id integer primary key,
        name text unique, price integer);
  insert into items (name,price) values ('Apple',800);                  
  insert into items (name,price) values ('Orange',500);
  insert into items (name,price) values ('Banana',300);  
""")
conn.commit()
#데이터 읽기
cur = conn.cursor() #문장실행 객체
#execute : sql 명령문 실행
cur.execute("select * from items")
#fetchall() : select 결과 전부를 리스트 전달
item_list = cur.fetchall()
print(item_list) #[(컬럼값1,컬럼값2,..),(...),()]
#반복문으로 조회
for id,name,price in item_list :
    print(id,name,price)

'''
  문제 : mydb sqlite 데이터 베이스 생성
       mydb에 member 테이블 생성하기
       id char(4) primary key, name char(15), email char(20) 인
       컬럼을 가진다.
'''
conn = sqlite3.connect("mydb")
cur = conn.cursor()
cur.execute("create table member \
            (id char(4) primary key, name char(15),email char(20))")
# 화면에서 id,이름,이메일를 입력받아 db에 등록하기
while True :
    d1 = input("사용자ID : ") #사용자아이디
    if d1 == '' :
       break 
    d2 = input("사용자이름 : ") #이름
    d3 = input("이메일 : ")     #이메일
    sql = "insert into member (id,name,email) values\
        ('"+d1+"','"+d2+"','"+d3+"')"
#insert into member (id,name,email) values ('kimsk','김삿갓','kim@aaa.bbb')        
    print(sql)
    cur.execute(sql) #실행.
    conn.commit()
conn.close()  #데이터베이스와 연결 종료. 실행시 다시 연결해야함

#파라미터를 이용하여 데이터 추가하기
conn = sqlite3.connect("mydb")
cur = conn.cursor()
while True :
    param=[] #?의 파라미터 값 사용
    d1 = input("사용자ID : ") #사용자아이디
    if d1 == '' :
       break 
    d2 = input("사용자이름 : ") #이름
    d3 = input("이메일 : ")     #이메일
    sql = "insert into member (id,name,email) values (?,?,?)"
    param.append(d1) #첫번째 등록. 첫번째 ?의 값
    param.append(d2) #두번째 등록. 두번째 ?의 값
    param.append(d3) #세번째 등록. 세번째 ?의 값
    cur.execute(sql,param)
    conn.commit()
conn.close()

#문제 : member 테이블의 내용을 출력하기
conn = sqlite3.connect("mydb")
cur = conn.cursor()
cur.execute("select * from member")
memlist = cur.fetchall(); #조회된 결과를 모두 리스트 리턴.
for m in memlist :
    print(m[0],m[1],m[2])
conn.close()    

#fetchone() 함수로 읽기
conn = sqlite3.connect("mydb")
cur = conn.cursor()
cur.execute("select * from member")
while True :
    row = cur.fetchone() #조회된 결과를 한개의 레코드씩 튜플로 리턴
    if row == None : #조회된 내용이 없는 경우
        break
    print(row)
conn.close()    

#executemany() : 여러개의 데이터를 한번에 추가하기
data=[('test7','테스트7','test7@aaa.bbb'),
      ('test8','테스트8','test8@aaa.bbb'),
      ('test9','테스트9','test9@aaa.bbb'),
      ('test10','테스트10','test10@aaa.bbb')]
conn = sqlite3.connect("mydb")
cur = conn.cursor()
cur.executemany\
    ("insert into member (id,name,email) values (?,?,?)",data)
conn.commit()
conn.close()    

#db 내용 수정하기
conn = sqlite3.connect("mydb")
cur = conn.cursor()

param = []
param.append("hongkd@aaa.bbb")
param.append("hongkd")
cur.execute("update member set email=? where id=?",param)
conn.commit()
conn.close()

# 이름이 테스트10 회원 정보 삭제하기
conn = sqlite3.connect("mydb")
cur = conn.cursor()
param = []
param.append("테스트10")
cur.execute("delete from member where name=?",param)
conn.commit()
conn.close()

#오라클 데이터 베이스에 접속하기
#오라클 모듈을 설정해야함
#pip install cx_Oracle #
#pip install : 외부모듈을 설정 명령어. console에서 실행
import cx_Oracle #오라클 접속을 위한 모듈. 기본 설정 아님
#connect("사용자아이디","비밀번호","서버IP/SID")
conn = cx_Oracle.connect('kic','1234','localhost/xe')
cur = conn.cursor() #sql 명령 객체

cur.execute("select * from student")
st_list = cur.fetchall()
for st in st_list :
  print(st)
conn.close()

'''
문제 : 학생테이블에
     학번(studno) : 5555, 이름(name):파이썬, 학년(grade):5,
     id : test1, jumin:9001011234567
     데이터 추가하기
''' 
import cx_Oracle 
conn = cx_Oracle.connect('kic','1234','localhost/xe')
cur = conn.cursor() #sql 명령 객체

sql = "insert into student (STUDNO,NAME,GRADE,ID,JUMIN) \
     values (:STUDNO,:NAME,:GRADE,:ID,:JUMIN)"  
cur.execute(sql,\
studno=5555,name='파이썬',grade=5,id='test1',jumin='9001011234567')
conn.commit()
#dictionary 객체
param={"studno":5557,"name":'파이썬3',"grade":5,"id":'test3',
       "jumin":'9001011234567'}    
cur.execute(sql,param)
conn.commit()
conn.close()

import cx_Oracle 
conn = cx_Oracle.connect('kic','1234','localhost/xe')
cur = conn.cursor() #sql 명령 객체

cur.execute("select * from student where grade=%d" % (5))
st_list = cur.fetchall()
for st in st_list :
  print(st)


    