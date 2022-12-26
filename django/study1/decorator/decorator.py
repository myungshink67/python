# -*- coding: utf-8 -*-
"""
Created on Fri Dec 23 09:53:36 2022

@author: KITCOOP
decorator.py : decorator 함수 모음
"""
from django.shortcuts import render

def loginIdchk(func):
    def check(request, id):
        try :
           login = request.session["login"]
        except : #로그아웃상태
           context = {"msg":"로그인하세요","url":"../../login"}
           return render(request,"alert.html",context)
        else :
           if login != id and login != 'admin' :
              context = {"msg":"본인만 가능합니다.","url":"../../main"}
              return render(request,"alert.html",context)
          
        return func(request, id)
    return check

def loginchk(func):
    def check(request):
        try :
           login = request.session["login"]
        except : #로그아웃상태
           context = {"msg":"로그인하세요","url":"../login"}
           return render(request,"alert.html",context)
        return func(request)
    return check


