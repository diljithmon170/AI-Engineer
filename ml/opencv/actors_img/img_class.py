import numpy as np
import cv2
import os
import matplotlib
from matplotlib import pyplot as plt

img = cv2.imread('./img/surya/s1.jpg')
img.shape
# plt.imshow(img)
# plt.show()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray.shape

face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('./haarcascade_eye.xml')
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

(x,y,w,h) = faces[0]

face_img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
plt.imshow(face_img)
plt.show()