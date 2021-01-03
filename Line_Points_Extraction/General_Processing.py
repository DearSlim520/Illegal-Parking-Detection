# -*- coding: utf-8 -*-
# @Time    : 2020/3/14 7:29 下午
# @Author  : Yingke Ding
# @FileName: General_Processing.py
# @Software: PyCharm

import cv2
import numpy as np


#img = cv2.imread("./Images/1733_cropped.jpg")
img = cv2.imread("./Images/IMG_1734_Lane_ROI.JPG")

img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # RGB 2 GRAY
img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)  # Extract HSV

lower_yellow = np.array([20, 100, 100], dtype="uint8")
upper_yellow = np.array([30, 255, 255], dtype="uint8")
mask_yellow = cv2.inRange(img_hsv, lower_yellow, upper_yellow)
mask_white = cv2.inRange(img_gray, 120, 255)
mask_yw = cv2.bitwise_or(mask_white, mask_yellow)
mask_yw_image = cv2.bitwise_and(img_gray, mask_yw)

kernel_size = 5
img_blurred = cv2.GaussianBlur(mask_yw_image, (kernel_size, kernel_size), 0)

cv2.imwrite("./Images/1734_test_roi.JPG", img_blurred)

# Show Image in IDLE
cv2.imshow("Image", img_blurred)
cv2.waitKey(0)
cv2.destroyAllWindows()



