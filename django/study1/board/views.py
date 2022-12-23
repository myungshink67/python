from django.shortcuts import render
from .models import Board  #현재 폴더의 models.py
from django.utils import timezone
from django.http import HttpResponseRedirect
from django.core.paginator import Paginator            

# Create your views here.
def write(request) :
    if request.method != "POST" :
      return render(request,"board/write.html")
    else :   #POST 방식 요청
      try :
         filename = request.FILES["file1"].name
         handle_upload(request.FILES["file1"]) 
      except :
         filename = "" 
      #num :    
      b=Board(name=request.POST["name"],\
              pass1=request.POST["pass"],\
              subject=request.POST["subject"],\
              content=request.POST["content"],\
              regdate=timezone.now(),\
              readcnt=0,file1=filename)   
      b.save() 
      return HttpResponseRedirect("../list")
      
def handle_upload(f) :
    with open("file/board/"+f.name,"wb") as dest :
        for ch in f.chunks() :
            dest.write(ch)
            
def list(request) :
    # pageNum 파라미터를 정수형으로 변환. 
    # 파라미터가 없으면 1이 기본값.
   pageNum = int(request.GET.get("pageNum",3))
   #모든 레코드 조회.
   # order_by("-num") : num 값의 내림차순 정렬.
   all_boards = Board.objects.all().order_by("-num")
   #Paginator : all_boards 목록을 10개씩 분리 저장.
   paginator = Paginator(all_boards,10)
   #paginator 객체에서 pageNum 번째 게시물 리턴
   #board_list : 페이지에 출력할 게시물 목록 저장
   board_list = paginator.get_page(pageNum)
   #등록된 게시물 건수
   listcount = Board.objects.count()
   return render(request,"board/list.html",\
         {"board":board_list,"listcount":listcount})
       
def info(request,num) :     
    board = Board.objects.get(num=num)
    board.readcnt += 1
    board.save()
    return render(request,"board/info.html",{"b":board})
    