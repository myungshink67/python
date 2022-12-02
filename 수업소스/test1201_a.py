# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 15:56:56 2022

@author: KITCOOP
test1201_a.py
"""
'''
1. dict_data 데이터를 이용하여 데이터프레임객체 df 생성하기
 단  index 이름은 r0,r1,r2로 설정
'''
import pandas as pd
dict_data = {'c0':[1,2,3], 'c1':[4,5,6], 'c2':[7,8,9], \
             'c3':[10,11,12], 'c4':[13,14,15]}
#1    
df = pd.DataFrame(dict_data, index=['r0', 'r1', 'r2'])
print(df)    

#2
df=pd.DataFrame(dict_data)
print(df)    
df.index=['r0','r1','r2'] 
print(df)    


'''
2. supplier_data.csv 파일을 
  pandas를 이용하여 읽고 Invoice Number,Cost,Purchase Date
  컬럼만 df_data.csv 파일에 저장하기
'''
import pandas as pd

infile = "data/supplier_data.csv"
df = pd.read_csv(infile)  #csv 형식의 파일읽기 

df_inset= df[["Invoice Number","Cost","Purchase Date"]]
print(df_inset)
#index=False : DataFrame 의 index는 파일로 저장안함 
df_inset.to_csv("data/df_data.csv",index=False) #DataFrame 객체를 csv 파일로 저장

"""
3. supplier_data.csv 파일을 
   pandas를 이용하여 읽고 Purchase Date 컬럼의 값이 1/20/14인 데이터만 
   140120_data.csv 파일로 저장하기
"""
import pandas as pd

infile = "data/supplier_data.csv"
df = pd.read_csv(infile)
print(df);

df_inset = df.loc[df["Purchase Date"] == "1/20/14"]
print(df_inset)
df_inset.to_csv("data/140120_data.csv",index=False)

'''
4. sales_2015.xlsx 파일의  january_2015 sheet의 중
   "Customer Name", "Sale Amount" 컬럼만 
   sales_2015_amt.xlsx 파일에 january_2015 sheet로 저장하기  
'''         
import pandas as pd
infile="data/sales_2015.xlsx" #원본파일
outfile = "data/sales_2015_amt.xlsx"  #목적파일.
#january_2015 sheet만 읽어 DataFrame 객체로 저장 
df = pd.read_excel(infile,"january_2015",index_col=None)
df_value = df[["Customer Name","Sale Amount"]]
print(df_value)
#엑셀파일로 저장
writer = pd.ExcelWriter(outfile)
df_value.to_excel(writer,sheet_name="january_2015",index=False)
writer.save() #엑셀파일 저장
writer.close()

'''
5. sales_2015.xlsx 파일의  모든 sheet의 
   "Customer Name", "Sale Amount" 컬럼만 
    sales_2015_allamt.xlsx 파일 각각의 sheet로 저장하기  
'''
import pandas as pd
infile="data/sales_2015.xlsx"
outfile = "data/sales_2015_allamt.xlsx"
writer = pd.ExcelWriter(outfile)  #엑셀파일
#sheet_name=None : 모든 sheet을 읽기
df = pd.read_excel(infile,sheet_name=None,index_col=None)
#df의 자료형 : 딕셔너리. {"sheet이름":해당 sheet의 데이터(DataFrame)}
for worksheet_name,data in df.items() :
    print("===",worksheet_name,"===")
    data_value = data[["Customer Name","Sale Amount"]]
    #data_value 값을 엑셀파일에 sheet추가하여 데이터저장
    data_value.to_excel(writer,sheet_name=worksheet_name,index=False)
writer.save() 
writer.close()

#원본파일의 모든 sheet를 하나의 sheet에 저장하기
import pandas as pd
infile="data/sales_2015.xlsx"
outfile = "data/sales_2015_oneamt.xlsx"
writer = pd.ExcelWriter(outfile)
df = pd.read_excel(infile,sheet_name=None,index_col=None)
row_output = []  #모든 sheet의 ["Customer Name","Sale Amount"] 컬럼만 저장
for worksheet_name,data in df.items() :
    print("===",worksheet_name,"===")
    data_value = data[["Customer Name","Sale Amount"]]
    row_output.append(data_value)
#row_output : [DataFrame1,DataFrame2,...]    
#pd.concat : DataFrame 객체 연결. 
# axis=0 : 행을 기준으로 연결
filtered_row = pd.concat(row_output,axis=0,ignore_index=True)    
filtered_row.to_excel\
    (writer,sheet_name="sales_2015_allamt",index=False)
writer.save() 
writer.close()
