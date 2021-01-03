# -*- coding: utf-8 -*-
# @Time    : 2020/3/12 10:29 下午
# @Author  : Yingke Ding
# @FileName: Merge_Lanenet_Processing.py
# @Software: PyCharm

from Image_Processing_Detect_Lines import *


binary = cv2.cvtColor((cv2.imread("./Images/lane2.jpeg")), cv2.COLOR_RGB2GRAY)
raw_img = cv2.imread("./Images/lane1_resized.jpeg")

# Gaussian Blur
kernel_size = 5
img_blurred = cv2.GaussianBlur(binary, (kernel_size, kernel_size), 0)

# Canny Edge Detection
low_threshold = 50
high_threshold = 150
img_canny = cv2.Canny(img_blurred, low_threshold, high_threshold)

# Add ROI Mask
img_shape = raw_img.shape
lower_left = [img_shape[1] / 9, img_shape[0]]
lower_right = [img_shape[1] - img_shape[1] / 9, img_shape[0]]
top_left = [img_shape[1] / 2 - img_shape[1] / 8, img_shape[0] / 2 + img_shape[0] / 10]
top_right = [img_shape[1] / 2 + img_shape[1] / 8, img_shape[0] / 2 + img_shape[0] / 10]
vertices = [np.array([lower_left, top_left, top_right, lower_right], dtype=np.int32)]
roi_image = region_of_interest(img_canny, vertices)

rho = 4
theta = np.pi / 180

threshold = 30
min_line_len = 100
max_line_gap = 180

line_image = hough_lines(binary, rho, theta, threshold, min_line_len, max_line_gap)
result = weighted_img(line_image, raw_img, alpha=0.8, beta=1., lamda=0.)

cv2.imshow("Image", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
