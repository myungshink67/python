from django.shortcuts import render

# Create your views here.
# member/views.py

#http://127.0.0.1:8000/member/login/ 요청시 호출되는 함수
def login(request) :
    return render(request,"member/login.html")
def join(request) :
    return render(request,"member/join.html")
