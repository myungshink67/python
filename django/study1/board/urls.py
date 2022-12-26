# -*- coding: utf-8 -*-
"""
Created on Fri Dec 23 11:08:13 2022

@author: KITCOOP
urls.py - board
"""
from django.urls import path
from . import views
urlpatterns=[
    path("write/",views.write,name="write"),
    path("list/",views.list,name="list"),
    path("info/<int:num>/",views.info,name="info"),
    path('update/<int:num>/', views.update,name='update'),
    path('delete/<int:num>/', views.delete, name='delete'),
]

