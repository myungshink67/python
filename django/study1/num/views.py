from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
import random
import base64
from django.http import JsonResponse

# Create your views here.
def index(request) :
    return render(request, template_name='num/index.html')

'''    
    data = request.FILES["file"]
    print()
    path = "static/num_images/"
    data = data[22:]
    filename = 'writing.png'
    image = open(path+filename, mode="wb")
    image.write(base64.b64decode(data))
    image.close()
#    answer = {"filename":"writing.png"}
#    return JsonResponse(answer)
    return 1;
'''

@csrf_exempt
def upload(request) :
    data = request.POST.__getitem__('data')
    data = data[22:]
    path = "static/num_images/"
    filename = 'mnist.png'
    image = open(path+filename,"wb")
    image.write(base64.b64decode(data))
    image.close()
#    answer = {"filename":"mnist.png"}
#    return JsonResponse(answer)
    return 1
