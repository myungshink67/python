# -*- coding: utf-8 -*-
"""
Created on Thu Dec 29 08:58:02 2022

@author: KITCOOP
20221229.py
"""
import cv2, numpy as np
face_cascade = cv2.CascadeClassifier\
                 ("haarcascade_frontalface_alt2.xml")  # 정면 검출기
eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml") #눈 검출기
image = cv2.imread('images/face1.jpg' , cv2.IMREAD_COLOR)
cv2.imshow("image",image)
image.shape #(349, 500, 3)
#cvtColor() : 이미지변환
#cv2.COLOR_BGR2GRAY : 컬러이미지 -> 흑백이미지.
#cv2.COLOR_BGR2HSV : 컬러이미지 -> 색상중 지배색상 변환.
#cv2.COLOR_BGR2RGB : 컬러이미지 -> 색상 반전
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 명암도 영상 변환
cv2.imshow("gray",gray)
gray.shape #(349, 500)

gray = cv2.equalizeHist(gray)  # 히스토그램 평활화
cv2.imshow("gray",gray)
#gray : 이미지값
#1.1 : 영상축소. 기본값 1.1
#2 : 최소 이웃되는 사각형값.
#(100, 100): 최소 객체의 크기.
faces = face_cascade.detectMultiScale\
         (gray, 1.1, 2, 0, (100, 100));  # 얼굴 검출
if faces.any():   #얼굴 사각형 검출?
    x, y, w, h = faces[0] #사각형의 좌표
    #face_image : 얼굴 영역
    face_image = image[y:y + h, x:x + w]  # 얼굴 영역 영상 가져오기
    cv2.imshow("face",face_image)
    eyes = eye_cascade.detectMultiScale\
                  (face_image, 1.15, 7, 0, (25, 20))  # 눈 검출 수행
    if len(eyes) == 2:  # 눈 사각형이 검출되면
        for ex, ey, ew, eh in eyes:
            center = (x + ex + ew // 2, y + ey + eh // 2)
            cv2.circle(image, center, 10, (0, 255, 0), 2)  # 눈 중심에 원 그리기
    else:
        print("눈 미검출")
    cv2.rectangle(image, faces[0], (255, 0, 0), 2)  # 얼굴 검출 사각형 그리기
    cv2.imshow("image", image)
else: print("얼굴 미검출")
cv2.waitKey(0)

