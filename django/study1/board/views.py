from django.shortcuts import render
from .models import Board  #현재 폴더의 models.py
from django.utils import timezone
from django.http import HttpResponseRedirect
from django.core.paginator import Paginator            
import traceback  #실행 예외의 메세지 출력

# Create your views here.
def write(request) :    #게시판 등록
    if request.method != "POST" :
      return render(request,"board/write.html")
    else :   #POST 방식 요청
      try :
         filename = request.FILES["file1"].name  #업로드 파일의 이름
         #request.FILES["file1"]:업로드 파일 내용
         handle_upload(request.FILES["file1"])  
      except : #업로드되는 파일이 없는 경우
         filename = "" 
      #num : 자동으로 1이 증가.   
      b=Board(name=request.POST["name"],\
              pass1=request.POST["pass"],\
              subject=request.POST["subject"],\
              content=request.POST["content"],\
              regdate=timezone.now(),\
              readcnt=0,file1=filename)   
      b.save() #db에 b객체의 값이 저장. insert실행.
               # num 기본키값이 새값
      return HttpResponseRedirect("../list")
      
def handle_upload(f) :
    #업로드위치 : BASE_DIR/file/board/ 폴더
    #f.name : 업로드 파일 이름
    with open("file/board/"+f.name,"wb") as dest :
        #f.chunks() : 업로드된 파일에서 버퍼만큼 읽기
        for ch in f.chunks() :
            dest.write(ch)  #출력파일에 저장
            
#게시물 목록 보기            
def list(request) :
    # pageNum 파라미터를 정수형으로 변환. 
    # 파라미터가 없으면 1이 기본값.
   pageNum = int(request.GET.get("pageNum",1))
   #모든 레코드 조회.
   # order_by("-num") : num 값의 내림차순 정렬.
   # all_boards : 등록된 전체 게시물 목록
   all_boards = Board.objects.all().order_by("-num")
   #Paginator : all_boards 목록을 10개씩 묶어 분리 저장.
   paginator = Paginator(all_boards,10)
   #paginator 객체에서 pageNum 번째 게시물 리턴
   #board_list : 페이지에 출력할 게시물 목록 저장
   board_list = paginator.get_page(pageNum)
   #등록된 게시물 건수
   listcount = Board.objects.count()
   return render(request,"board/list.html",\
         {"board":board_list,"listcount":listcount})
       
#num : url에 표시된 게시물 번호.       
def info(request,num) :     
    board = Board.objects.get(num=num)  #num 값에 해당하는 게시물 한개 저장
    board.readcnt += 1  #조회건수값 1증가
    board.save()        #db에 조회건수 저장
    return render(request,"board/info.html",{"b":board})
    
def update(request, num):
    if request.method != 'POST':
        board = Board.objects.get(num=num)
        return render(request, 'board/update.html',{'b': board})
#  1. 비밀번호 검증.
#     오류 : 비밀번호 오류 메세지 출력. update 화면 페이지 이동
#  2. 내용 수정
#     수정완료 : 상세보기 페이지 이동
#     수정실패 : 메세지 출력 후 update 화면 페이지 이동
    else :
        board = Board.objects.get(num=num)
        pass1 = request.POST["pass"]
        if board.pass1 != pass1 :
            context = {"msg":"비밀번호 오류","url":"../../update/"+str(num) + "/"}
            return render(request,"alert.html",context)
        try :
            filename = request.FILES["file1"].name
            handle_upload(request.FILES["file1"])
        except :
            filename = ""
        try :
            if filename == "" :
               filename = request.POST["file2"]
            b=Board(num=num,\
                    name=request.POST["name"],\
                    pass1=request.POST["pass"],\
                    subject=request.POST["subject"],\
                    content=request.POST["content"],\
                    file1=filename)   
            b.save()
            return HttpResponseRedirect("../../list/")
        except Exception as e :
            print(traceback.format_exc()) #runserver 콘솔창에 오류 메세지 출력
            context = {"msg":"게시물 수정 실패",\
                       "url":"../../update/"+str(num) + "/"}
            return render(request,"alert.html",context)
            
# 게시물 삭제하기
def delete (request, num):
    if request.method != 'POST':
        return render(request, 'board/delete.html',{"num":num})
    else :
# 1. 비밀번호 검증
#    오류 : 비밀번호 오류 메세지 출력. delete 페이지 출력
# 2. 해당 게시물 삭제.
#    게시물 목록(list) 페이지 출력        
       b=Board.objects.get(num=num)
       pass1 = request.POST["pass"]
       if pass1 != b.pass1 :
          context = {"msg":"비밀번호 오류",\
                       "url":"../../delete/"+str(num) + "/"}
          return render(request,"alert.html",context)
       try :
          b.delete()
          return HttpResponseRedirect("../../list/")
       except :
          print(traceback.format_exc()) #runserver 콘솔창에 오류 메세지 출력
          context = {"msg":"게시물 삭제 실패",\
                       "url":"../../delete/"+str(num) + "/"}
          return render(request,"alert.html",context)
          
           
