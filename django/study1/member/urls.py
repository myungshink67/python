# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 10:19:40 2022

@author: KITCOOP
member/urls.py
"""

from django.urls import path
from . import views
# http://127.0.0.1:8000/member/
urlpatterns=[
    #http://127.0.0.1:8000/member/login/ 요청시 view.py
    #  의 login 함수 실행
    path("login/",views.login, name="login"),
]