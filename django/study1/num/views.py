#  num/view.py
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
import base64
from django.http import JsonResponse
from . import knn

def index(request) :
    return render\
        (request, template_name='num/index.html')
        
# ajax으로 파일 업로드.        
@csrf_exempt   #csrf 파라미터 전송이 없어도 가능.
def upload(request) :
#      data = request.POST.__getitem__('data')  #data:캔버스에서 그려진 이미지파일
      data = request.POST['data']  #data:캔버스에서 그려진 이미지파일
      data = data[22:] #header 부분 제거
      #static/num_images/ : 폴더 생성.
      #train_numbers.png 파일 복사
      path = "static/num_images/"
      filename = 'mnist.png'
      image = open(path+filename,"wb")
      image.write(base64.b64decode(data)) #base64형태의 데이터를 이진형태 변경
      image.close()
      answer = {"result":knn.prednum()}
      return JsonResponse(answer) #json 객체로 클라이언트로 전달
