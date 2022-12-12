import time
from selenium import webdriver
#네이버 로그인 하기
driver = webdriver.Chrome("C:/20210416/setup/chromedriver")
time.sleep(1)
driver.get("https://nid.naver.com/nidlogin.login?mode=form&url=https%3A%2F%2Fwww.naver.com")
id = input("네이버 아이디를 입력하세요 : ")
# <input ... name='id' ... >
driver.execute_script\
    ("document.getElementsByName('id')[0].value='"+id+"'")
pw = input("네이버 비밀번호를 입력하세요 : ")
# <input ... name='pw' ... >
driver.execute_script\
    ("document.getElementsByName('pw')[0].value='"+pw+"'")
# 로그인 버튼 클릭
# //*[@id="log.login"] : <... id="log.login" ...>    
driver.find_element_by_xpath('//*[@id="log.login"]').click()
time.sleep(1)
# 쇼핑 영역 클릭하기
driver.find_element_by_xpath\
    ('//*[@id="NM_FAVORITE"]/div[1]/ul[1]/li[5]/a').click()
time.sleep(1)
# My쇼핑 영역 클릭하기
driver.find_element_by_xpath\
                   ('//*[@id="_myPageWrapper"]/a').click()
time.sleep(1)
#주문확인/배송조회 클릭하기
driver.find_element_by_xpath\
 ('//*[@id="_myPageWrapper"]/div/div[3]/ul[2]/li[2]/a').click()
time.sleep(1)
#주문정보 보기
#find_elements_by_css_selector : 선택자로 태그 객체들 지정
# .goods_pay_section  : <... class="goods_pay_section" ...>
products = driver.find_elements_by_css_selector\
    (".goods_pay_section")
for product in products:
    print("-", product.text)
time.sleep(2)   
driver.quit()