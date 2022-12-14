# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 16:20:02 2022
@author: KITCOOP
test1213_a.py
"""

'''
1. chipotle.tsv 파일을 읽고 item 별 판매 갯수 시각화하기.
   가장 많이 판매한 상품 10개만 막대그래프로 출력하기
   20221213-1.png 참조
'''   

import pandas as pd
import matplotlib.pyplot as plt
chipo = pd.read_csv("data/chipotle.tsv", sep = '\t')

chipo_chicken = chipo[chipo['item_name'] == "Chicken Bowl"]
chipo_chicken["quantity"].sum()

#주문상품별 판매 수량
item_qty = chipo.groupby("item_name")["quantity"].sum()
item_qty
# 판매수량의 내림차순 정렬. 수량이 많은 10개 선택
item_qty = item_qty.sort_values(ascending=False)[:10]
item_qty

#상품명
item_name_list = item_qty.index.tolist()
#판매 수량
sell_cnt = item_qty.values.tolist()

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(1,1,1) 
ax.bar(item_name_list, sell_cnt, align='center')
plt.ylabel('item_sell_count')
plt.xlabel('item Name')
ax.set_xticklabels(item_name_list, rotation=90)
plt.title('Distribution of all sell item')
plt.show()
plt.savefig("20221213-1.png",dpi=400,bbox_inches="tight")


'''
2. chipotle.tsv 파일을 읽고
   Chicken Bowl을 2개 이상 주문한 주문 횟수 구하기
   주문번호    Chicken Bowl 주문수량
       1                 2
       2                 3
       3                 1
    주문횟수 :   2    1,2,번주문만 횟수 
'''
import pandas as pd
chipo = pd.read_csv("data/chipotle.tsv", sep = '\t')
#chipo_chicken : 주문상품이 Chicken Bowl인 데이터 목록
chipo_chicken = chipo[chipo['item_name'] == "Chicken Bowl"]
chipo_chicken
#주문수량이 2개 이상인 데이터 저장
chipo_chicken_result = \
    chipo_chicken[chipo_chicken['quantity'] >= 2]
chipo_chicken_result    
chipo_chicken_result["quantity"]
print(chipo_chicken_result.shape[0]) #행의 수
print(len(chipo_chicken_result)) #행의 수


