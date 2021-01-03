# !/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2020/3/30
# @Author  : Yingke Ding
# @File    : car_outline.py
# @Software: PyCharm

import cv2
import numpy as np


img = cv2.imread("./Images/IMG_1731.JPG")

image_cropped = img[1200:1500, 1200:2750]  # 3840 2160

cv2.imwrite("./Images/1731_cropped_car.JPG", image_cropped)

img = image_cropped

kernel_size = 5
img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

low_threshold = 50
high_threshold = 150
img = cv2.Canny(img, low_threshold, high_threshold)

x = cv2.Sobel(img, cv2.CV_16S, 1, 0)
y = cv2.Sobel(img, cv2.CV_16S, 0, 1)

absX = cv2.convertScaleAbs(x)
absY = cv2.convertScaleAbs(y)

dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)

cv2.imshow("absX", absX)
cv2.imshow("absY", absY)

cv2.imshow("Result", dst)

cv2.imwrite("./Images/1731_car.JPG", absY)

cv2.waitKey(0)
cv2.destroyAllWindows()
