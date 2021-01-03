# -*- coding: utf-8 -*-
# @Time    : 2020/3/12 11:01 下午
# @Author  : Yingke Ding
# @FileName: Image_Resize.py
# @Software: PyCharm

import cv2


image = cv2.imread("./Images/lane1.jpeg")
image = cv2.resize(image, (1024, 512), interpolation=cv2.INTER_LINEAR)

cv2.imwrite("./Images/lane1_resized.jpeg", image)