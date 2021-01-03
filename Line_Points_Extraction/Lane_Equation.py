# -*- coding: utf-8 -*-
# @Time    : 2020/3/14 8:22 下午
# @Author  : Yingke Ding
# @FileName: Lane_Equation.py
# @Software: PyCharm

import cv2
import numpy as np


def main(img_dir):
    """
    Get the line equation. Only for a single straight line.
    ROI_LANE_IMAGE -> CANNY -> HOUGH -> A MEAN LINE
    :param img_dir:
    :return:
    """
    img = cv2.imread(img_dir)
    img_test = cv2.imread("./Images/IMG_1735.JPG")

    print(img.shape)

    # img -> Canny = dst
    dst = cv2.Canny(img, 50, 200, None, 3)

    # dst -> Hough = lines
    lines = cv2.HoughLinesP(image=dst, rho=1, theta=np.pi / 180, threshold=50, lines=None,
                            minLineLength=50, maxLineGap=10)

    # Plot the result lines on raw img.
    if lines is not None:
        for i in range(0, len(lines)):
            l = lines[i][0]
            cv2.line(img, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 3, cv2.LINE_AA)

    slopes = []
    intercepts = []
    y_min = lines[0][0][1]  # lines: [[[x1, y1, x2, y2]], [[...]], ...]
    y_max = lines[0][0][1]

    for line in lines:
        for x1, y1, x2, y2 in line:
            if y1 < y_min or y2 < y_min:
                y_min = min(y1, y2)
            if y1 > y_max or y2 > y_max:
                y_max = max(y1, y2)

            line_slope = (y2 - y1) / (x2 - x1)
            line_intercept = np.mean([y2 - line_slope * x2, y1 - line_slope * x1])
            slopes.append(line_slope)
            intercepts.append(line_intercept)

    slope_mean = np.mean(slopes, axis=0)
    intercepts_mean = np.mean(intercepts, axis=0)

    x_min = (y_min - intercepts_mean) / slope_mean
    x_max = (y_max - intercepts_mean) / slope_mean

    point1 = (int(x_min), int(y_min))
    point2 = (int(x_max), int(y_max))

    # Plot the average line on raw img.
    color = [0, 255, 0]  # Green
    thickness = 6
    cv2.line(img, point1, point2, color=color, thickness=thickness)

    cv2.imwrite("./Images/1733_RESULT.JPG", img)


if __name__ == '__main__':
    main(img_dir="./Images/1733_Extracted.JPG")
