# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 15:49:01 2022

@author: KITCOOP

test1201.py
"""

'''
1. dict_data 데이터를 이용하여 데이터프레임객체 df 생성하기
 단  index 이름은 r0,r1,r2로 설정
'''
dict_data = {'c0':[1,2,3], 'c1':[4,5,6], 'c2':[7,8,9], \
             'c3':[10,11,12], 'c4':[13,14,15]}


'''
2. supplier_data.csv 파일을 
  pandas를 이용하여 읽고 Invoice Number,Cost,Purchase Date
  컬럼만 df_data.csv 파일에 저장하기
'''

"""
3. supplier_data.csv 파일을 
   pandas를 이용하여 읽고 Purchase Date 컬럼의 값이 1/20/14인 데이터만 
   140120_data.csv 파일로 저장하기
"""

'''
4. sales_2015.xlsx 파일의  january_2015 sheet의 중
   "Customer Name", "Sale Amount" 컬럼만 
   sales_2015_amt.xlsx 파일에 january_2015 sheet로 저장하기  
'''         

'''
5. sales_2015.xlsx 파일의  모든 sheet의 
   "Customer Name", "Sale Amount" 컬럼만 
    sales_2015_allamt.xlsx 파일 각각의 sheet로 저장하기  
'''
