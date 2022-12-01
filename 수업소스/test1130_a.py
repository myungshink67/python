# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 16:20:41 2022

@author: KITCOOP
test1130_a.py
"""


'''
1. mod2.py 파일을 읽어서 mod2.bak 파일로 복사하기.
'''
infp = open("mod2.py", "r", encoding='UTF8') #원본파일
outfp = open("mod2.bak","w",encoding="UTF8") #복사본파일
while True :
    inStr = infp.readline() #text 형태의 파일 가능 
    if inStr == '' : #EOF(End Of File. 파일의 끝)
        break
    outfp.writelines(inStr)
infp.close()
outfp.close()
print("프로그램 종료")

'''
2. 현재폴더에 temp폴더를 생성하고, 생성된 폴더에 indata.txt 파일을
   생성하여 생성된파일에 키보드에서 입력된 정보 저장하는 프로그램 
   구현하기
'''
import os
wpath = os.getcwd()+"/temp"
if  not os.path.exists(wpath) :
    os.mkdir("temp") #temp 폴더 생성
   
outfp = open("temp/indata.txt",'w',encoding='UTF8')
while True :
    inline = input("저장 내용=>")
    if inline == '' :
        break
    outfp.writelines(inline+"\n")
outfp.close()   

'''
3.원본 파일의 이름을 입력받고, 입력받은 복사본 파일이름으로 복사하는 
  프로그램 작성하기
  원본파일이 없는 경우 원본 파일이 존재하지 않습니다. 출력하기

[결과]
원본파일의 이름을 입력하세요 : aaa
원본파일이 존재하지 않습니다.

원본파일의 이름을 입력하세요 : data.txt
복사본파일의 이름을 입력하세요 : databak.txt
복사완료
'''
#inFp=open('aaa', "r",  encoding='utf-8')

infile = input("원본파일의 이름을 입력하세요 : ")
try :
     #읽기 위한 원본파일은 존재하지 않으면 예외 발생
     inFp=open(infile, "r",  encoding='utf-8')
except :
     print("원본파일이 존재하지 않습니다.")   
else :     
     outfile = input("복사본파일의 이름을 입력하세요 : ")
     outFp=open(outfile, "w",  encoding='utf-8')
     #readlines() : 모든 라인을 읽어서 리스트로 리턴 함수 
     inList = inFp.readlines() 
     for inStr in inList :
          outFp.writelines(inStr)
     inFp.close()
     outFp.close()
     print("\n복사완료")

'''
 4. 다음 number 데이터를 이용하여 큰글씨를 출력하는 프로그램 작성하기 
   [결과]
   숫자를 입력하세요 =>12345
     *  ***  ***  * *  ***  
     *    *    *  * *  *    
     *  ***  ***  ***  ***  
     *  *      *    *    *  
     *  ***  ***    *  ***  
'''
number= [["*** ","* * ","* * ","* * ","*** "],  #0
	     ["  * ","  * ","  * ","  * ","  * "],  #1
		 ["*** ","  * ","*** ","*   ","*** "],
		 ["*** ","  * ","*** ","  * ","*** "],
		 ["* * ","* * ","*** ","  * ","  * "],
		 ["*** ","*   ","*** ","  * ","*** "],
		 ["*** ","*   ","*** ","* * ","*** "],
		 ["*** ","  * ","  * ","  * ","  * "],
		 ["*** ","* * ","*** ","* * ","*** "],
		 ["*** ","* * ","*** ","  * ","  * "]
		]

num = input("숫자를 입력하세요 =>") #1234
for i in range(5) : #i : 0~4
    for n in num : #4
        print(number[int(n)][i],end=" ")
    print()    

'''
 5. sqlite의 mydb의 테이블의 내용 조회하기
 sql 구문을 입력하고 다음과 같은 결과가 출력되도록 프로그램을 작성하시오
[결과]
sql 입력하세요=========
select * from member

조회 레코드수: 6 ,조회 컬럼수: 3

('hongkd', '홍길동', 'hongkd@aaa.bbb')
('kimsg', '김삿갓', 'kimsg@aaa.bbb')
('test1', '테스트1', 'test1@aaa.bbb')
('test2', '테스트2', 'test2@aaa.bbb')
('test3', '테스트3', 'test3@aaa.bbb')
('test4', '테스트4', 'test4@aaa.bbb')
sql 입력하세요=========
insert into member (id,name,email) values ('a','a','a')
sql 입력하세요=========
select * from member

조회 레코드수: 7 ,조회 컬럼수: 3

('hongkd', '홍길동', 'hongkd@aaa.bbb')
('kimsg', '김삿갓', 'kimsg@aaa.bbb')
('test1', '테스트1', 'test1@aaa.bbb')
('test2', '테스트2', 'test2@aaa.bbb')
('test3', '테스트3', 'test3@aaa.bbb')
('test4', '테스트4', 'test4@aaa.bbb')
('a', 'a', 'a')
sql 입력하세요=========

'''
import sqlite3
conn = sqlite3.connect("mydb") 
cur = conn.cursor()
while True :
    sql = input("sql 입력하세요=========\n")
    if sql=="" :
        break
    cur.execute(sql)
    rows = cur.fetchall()
    if len(rows) > 0 : #select 구문인 경우
       print()
       print("조회 레코드수:",len(rows),",조회 컬럼수:",len(rows[0]))    
       print()
       for row in rows :
           print(row)
cur.close()
conn.close();
