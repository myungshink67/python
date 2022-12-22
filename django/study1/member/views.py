from django.shortcuts import render
from .models import Member
from django.http import HttpResponseRedirect
from django.contrib import auth
import time
# member/views.py
#http://127.0.0.1:8000/member/login/ 요청시 호출되는 함수
def login(request) :
    print("1:",request.session.session_key)
    if request.method != "POST" :
       return render(request,"member/login.html")
    else :
       id1=request.POST["id"] 
       pass1=request.POST["pass"]
       try :
           #입력된 id값으로 Member 객체에서 조회
           member = Member.objects.get(id=id1) #select 문장 실행
       except :  #db에 아이디 정보가 없을 때
           context = {"msg":"아이디를 확인하세요."}
           return render(request,"member/login.html",context)
       else :  #정상적인 경우. 아이디 정보가 조회된 경우
           #member.pass1 : db에 등록된 비밀번호
           #pass1 : 입력된 비밀번호
           if member.pass1 == pass1 :  #로그인 정상
              time.sleep(1)
              print("2:",request.session.session_key)
              request.session["login"] = id1  #session 객체에 login 등록.
              return HttpResponseRedirect("../main")
           else :  #비밀번호 오류
               context = {"msg":"비밀번호가 틀립니다.","url":"../login/"}
               return render(request,"alert.html",context)
               

def join(request) :
    if request.method != "POST" :
       return render(request,"member/join.html")
    else : #POST 방식. 
       #request.POST["id"] : id 파라미터 값.
       member = Member(id=request.POST["id"],\
                    pass1=request.POST["pass"],\
                    name=request.POST["name"],
                    gender=request.POST["gender"],
                    tel=request.POST["tel"],
                    email=request.POST["email"],
                    picture=request.POST["picture"]) 
       member.save() #insert 문장 실행.
       return HttpResponseRedirect("../login/")

def main(request):
   print("3",request.session.session_key)
   return render(request, 'member/main.html')

def logout(request) :
    print(request.session.session_key)
    auth.logout(request) #세션종료
    return HttpResponseRedirect("../login/")
    
# urls.py : info/<std:id>/ => info요청 url에서 id 값을 id
#                             매개변수로 전달 
def info(request,id) :
    try :
      login = request.session["login"]
    except : #로그아웃상태
      context = {"msg":"로그인하세요","url":"../../login"}
      return render(request,"alert.html",context)
    else : #로그인 상태
      if login == id or login == 'admin' :
          member = Member.objects.get(id=id)
          return render(request,"member/info.html",{"mem":member})
      else :
        context = {"msg":"본인만 가능합니다.","url":"../../main"}
        return render(request,"alert.html",context)
    
def update(request,id) :
    try :
      login = request.session["login"]
    except : #로그아웃상태
      context = {"msg":"로그인하세요","url":"../../login"}
      return render(request,"alert.html",context)
    else : #로그인 상태
      if login == id or login == 'admin' :
         return update_rtn(request,id)
      else :
        context = {"msg":"본인만 가능합니다.","url":"../../main"}
        return render(request,"alert.html",context)
    
def update_rtn(request,id) :
    member = Member.objects.get(id=id)
    if request.method != "POST" :
       return render(request,"member/update.html",{"mem":member})
    else :
       #비밀번호 검증. 
       #비밀번호 오류시 비밀번호 오류 메세지, update.html 페이지 출력
       #member.pass1 : db에 등록된 비밀번호
       # request.POST["pass"] : 입력된 비밀번호
       if request.POST["pass"] == member.pass1:
          member = Member(id=request.POST["id"],\
                    pass1=request.POST["pass"],\
                    name=request.POST["name"],
                    gender=request.POST["gender"],
                    tel=request.POST["tel"],
                    email=request.POST["email"],
                    picture=request.POST["picture"]) 
          #id값존재:update, id값없으면 insert    
          member.save() #update 문장 실행.
          #member.delete() #db에서 삭제
          return HttpResponseRedirect("../../info/"+id+"/")
       else :
        context = {"msg":"비밀번호 오류입니다.",\
                   "url":"../../update/"+id+"/"}
        return render(request,"alert.html",context)
           
def delete(request,id) :
    try :
        login = request.session["login"]
    except :
      context = {"msg":"로그인하세요","url":"../../login"}
      return render(request,"alert.html",context)
    else :
      if login == id or login == 'admin' :
         return delete_rtn(request,id)
      else :
        context = {"msg":"본인만 가능합니다.",\
                   "url":"../../main"}
        return render(request,"alert.html",context)

def delete_rtn(request,id) :
   if request.method != "POST" :
      return render(request,"member/delete.html",{"id":id})
   else :
      login = request.session["login"]
      member = Member.objects.get(id=login)       
      if member.pass1 == request.POST["pass"] : #비밀번호 일치
         mem = Member.objects.get(id=id)
         mem.delete()
         if id == login : #본인탈퇴
             auth.logout(request) #로그아웃
             context={"msg":"탈퇴완료","url":"../../login/"}
             return render(request,"alert.html",context)
         else : #관리자 강제탈퇴
             return HttpResponseRedirect("../../list/")
      else :      #비밀번호 불일치    
         context={"msg":"비밀번호 오류",\
                  "url":"../../delete/"+id+"/"}
         return render(request,"alert.html",context)

def list(request) :
    try :
        login = request.session["login"]
    except :
        context={"msg":"로그인 하세요","url":"../login/"}
        return render(request,"alert.html",context)
    else :
        if login != "admin" :
           context={"msg":"관리자만 가능합니다","url":"../main/"}
           return render(request,"alert.html",context)
        #mlist 요소: Member 객체 
        mlist = Member.objects.all() #모든데이터 리턴
        return render(request,"member/list.html",{"mlist":mlist})

def picture(request) :
    if request.method != 'POST':
       return render(request, 'member/pictureform.html')
    else :
        #request.FILES["picture"] : 업로드된 파일 객체
        #fname : 파일이름
       fname = request.FILES["picture"].name
       handle_upload(request.FILES["picture"])
       return render(request,"member/picture.html",{"fname":fname})

def handle_upload(f) :
    with open("file/picture/"+f.name,"wb") as dest : #close 생략 가능
        #f.chunks() : 이진파일 읽기.
        for ch in f.chunks() :
            dest.write(ch)
            