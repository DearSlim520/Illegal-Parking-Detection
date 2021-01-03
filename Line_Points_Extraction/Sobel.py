# !/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2020/4/2
# @Author  : Yingke Ding
# @File    : Sobel.py
# @Software: PyCharm

import cv2
import numpy as np

img = cv2.imread("/Users/ding/Desktop/IMG_1869.jpg")
img = cv2.resize(img, (2000, 1503), 0, 0)

x = cv2.Sobel(img, cv2.CV_16S, 1, 0)
y = cv2.Sobel(img, cv2.CV_16S, 0, 1)

absX = cv2.convertScaleAbs(x)
absY = cv2.convertScaleAbs(y)

dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)

cv2.imshow("absX", absX)
cv2.imshow("absY", absY)

#cv2.imshow("Result", dst)

cv2.waitKey(0)
cv2.destroyAllWindows()
