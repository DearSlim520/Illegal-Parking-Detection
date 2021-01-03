# -*- coding: utf-8 -*-
# @Time    : 2020/3/14 3:17 下午
# @Author  : Yingke Ding
# @FileName: Gaussian_Blur.py
# @Software: PyCharm

import cv2


image = cv2.imread("./Images/1731_cropped.JPG")

kernel_size = 5
img_blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

cv2.imwrite("./Images/1731_cropped_blurred.JPG", img_blurred)
