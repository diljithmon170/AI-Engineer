import numpy as np
import cv2
import os
import matplotlib
from matplotlib import pyplot as plt

img = cv2.imread('./img/surya/s1.jpg')
img.shape
plt.imshow(img)
plt.show()