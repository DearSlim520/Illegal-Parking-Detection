# -*- coding: utf-8 -*-
# @Time    : 2020/3/12 10:48 下午
# @Author  : Yingke Ding
# @FileName: Image_Crop.py
# @Software: PyCharm

import cv2


image = cv2.imread("./Images/IMG_2996.jpg")
image_cropped = image[2300:3024, 500:2200]

cv2.imwrite("./Images/2996_cropped.jpg", image_cropped)
