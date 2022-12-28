# -*- coding: utf-8 -*-
# num/knn.py

import numpy as np , cv2
import matplotlib.pyplot as plt
from study1 import settings

def find_value_position(img, direct):
    project = cv2.reduce(img, direct, cv2.REDUCE_AVG).ravel()
    p0, p1 = -1, -1
    len = project.shape[0]   #전체 길이
    for i in range(len):
        if p0 < 0 and project[i] < 250: p0 = i  #시작좌표
        if p1 < 0 and project[len-i-1] < 250 : p1 = len-i-1 #종료좌표
    return p0, p1  #시작좌표와 종료 좌표


def find_number(part): #숫자이미지
    x0, x1 = find_value_position(part, 0)
    y0, y1 = find_value_position(part, 1)
    return part[y0:y1, x0:x1]

#1. 이미지를 가운데 부분 배치
#2. 크기를 (40,40) 변경
def place_middle(number, new_size):
    h, w = number.shape[:2]
    big = max(h, w)
    square = np.full((big, big), 255, np.float32) #모든 값을 255로 채움
    dx, dy = np.subtract(big, (w,h))//2
    square[dy:dy + h, dx:dx + w] = number
    return cv2.resize(square, new_size).flatten()

#views.py에서 호출
def prednum() :
    test_img = cv2.imread(str(settings.BASE_DIR) +\
              '/static/num_images/mnist.png', cv2.IMREAD_GRAYSCALE)
    cv2.threshold(test_img, 32, 255, cv2.THRESH_BINARY)
    num = find_number(test_img)
    data = place_middle(num, size)
    data = data.reshape(1, -1) 
    _, resp, _, _ = knn.findNearest(data, 5) #data 예측하기. 이웃갯수 5로 지정
    return '결과 : ' + str(int(resp[0][0]))


# train_numbers.png 이미지를 읽어서, 학습데이터로 사용
# 학습 하기.
size= (40, 40)   #숫자데이터 크기.
nclass, nsample = 10, 20   

#흑백으로 이미지 읽기
train_image = cv2.imread(str(settings.BASE_DIR) + \
        '/static/num_images/train_numbers.png', cv2.IMREAD_GRAYSCALE)
train_image = train_image[5:405, 6:806] #여백 제거. 
cv2.threshold(train_image, 32, 255, cv2.THRESH_BINARY)
cells = [np.hsplit(row, nsample) for row in np.vsplit(train_image, nclass)]

nums = [find_number(c) for c in np.reshape(cells, (-1, 40, 40))]
len(nums)
trainData = np.array([place_middle(n, size) for n in nums])
labels= np.array([ i for i in range(nclass) for j in range(nsample)], np.float32)

#knn 알고리즘으로 학습
knn = cv2.ml.KNearest_create()
knn.train(trainData,cv2.ml.ROW_SAMPLE,labels) #학습하기



