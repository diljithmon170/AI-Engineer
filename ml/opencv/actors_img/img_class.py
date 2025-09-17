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

face_cascade = cv2.CascadeClassifier('./opencv/haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('./opencv/haarcascades/haarcascade_eye.xml')
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

(x,y,w,h) = faces[0]

face_img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
# plt.imshow(face_img)
# plt.show()


for (x,y,w,h) in faces:
    face_img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = face_img[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
# plt.figure()
# plt.imshow(face_img, cmap='gray')
# plt.show()


# prepocessing crop the facial part
# plt.imshow(roi_color, cmap='gray')
# plt.show()

cropped_img = np.array(roi_color)

# 3. prepocessing - Use wavelet transform as a feature for traning our model

import pywt
def w2d(img, mode='haar', level=1):
    imArray = img
    # convert to grayscale
    if len(imArray.shape) == 3:
        imArray = cv2.cvtColor(imArray, cv2.COLOR_BGR2GRAY)
    imArray = np.float32(imArray)
    imArray /= 255
    coeffs = pywt.wavedec2(imArray, mode, level=level)
    coeffs_H = list(coeffs)  
    coeffs_H[0] *= 0  
    imArray_H = pywt.waverec2(coeffs_H, mode)  
    imArray_H *= 255
    imArray_H = np.uint8(imArray_H)
    return imArray_H

im_har = w2d(cropped_img, 'db1', 5)
# plt.imshow(im_har, cmap='gray')
# plt.show()

# 3. Preprocessing: Load image, detect face. If eyes >=2, then save and crop the face region

# Lets write a python function that can take input image and returns cropped image (if face and eyes >=2 are detected)

def get_cropped_image_if_2_eyes(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        if len(eyes) >= 2:
            return roi_color

original_image = cv2.imread('./img/surya/s2.jpg')
# plt.imshow(original_image)
# plt.show()

cropped_image = get_cropped_image_if_2_eyes('./img/surya/s2.jpg')
# plt.imshow(cropped_image)
# plt.show()

path_to_data = './img/'
path_to_cr_data = './img/cropped/'
"""
img_dirs = []
for entry in os.scandir(path_to_data):
    if entry.is_dir():
        img_dirs.append(entry.path)

# Go through all images in dataset folder and create cropped images for them. There will be cropped folder inside dataset folder after you run this code

import shutil
if os.path.exists(path_to_cr_data):
    shutil.rmtree(path_to_cr_data)
os.mkdir(path_to_cr_data)

cropped_image_dirs = []
celebrity_file_names_dict = {}
for img_dir in img_dirs:
    count = 1
    celebrity_name = img_dir.split('/')[-1]
    celebrity_file_names_dict[celebrity_name] = []
    for entry in os.scandir(img_dir):
        if entry.is_file():
            image_path = entry.path
            cropped_image = get_cropped_image_if_2_eyes(image_path)
            if cropped_image is not None:
                cropped_folder = path_to_cr_data + celebrity_name
                if not os.path.exists(cropped_folder):
                    os.mkdir(cropped_folder)
                    cropped_image_dirs.append(cropped_folder)
                cropped_file_name = celebrity_name + str(count) + '.jpg'
                cropped_file_path = cropped_folder + '/' + cropped_file_name
                cv2.imwrite(cropped_file_path, cropped_image)
                celebrity_file_names_dict[celebrity_name].append(cropped_file_path)
                count += 1
 # Print the dictionary to see the results
"""
# Manually examine cropped folder and delete any unwanted images

celebrity_file_names_dict = {}
for img_dir in cropped_image_dirs:
    celebrity_name = img_dir.split('/')[-1]
    file_list = []
    for entry in os.scandir(img_dir):
        file_list.append(entry.path)
    celebrity_file_names_dict[celebrity_name] = file_list