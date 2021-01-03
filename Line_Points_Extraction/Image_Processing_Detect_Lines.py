# -*- coding: utf-8 -*-
# @Time    : 2020/3/12 4:25 下午
# @Author  : Yingke Ding
# @FileName: Image_Processing_Detect_Lines.py
# @Software: PyCharm

import numpy as np
import cv2


def region_of_interest(image, vertices):
    """
    Applies an image mask Region Of Interest (ROI).

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    :param image:
    :param vertices:
    :return:
    """

    # defining a blank mask to start with
    mask = np.zeros_like(image)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(image.shape) > 2:
        channel_count = image.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image


def get_slope(x1, y1, x2, y2):
    return (y2 - y1) / (x2 - x1)


# thick red lines
def draw_lines(img, lines, color=[255, 0, 0], thickness=6):
    """
    workflow:
    1) examine each individual line returned by hough & determine if it's in left or right lane by its slope
    because we are working "upside down" with the array, the left lane will have a negative slope and right positive
    2) track extrema
    3) compute averages
    4) solve for b intercept
    5) use extrema to solve for points
    6) smooth frames and cache
    """

    global cache
    first_frame = 1
    y_global_min = img.shape[0]  # min will be the "highest" y value, or point down the road away from car
    y_max = img.shape[0]
    l_slope, r_slope = [], []
    l_lane, r_lane = [], []
    det_slope = 0.4
    alpha = 0.2
    # i got this alpha value off of the forums for the weighting between frames.
    # i understand what it does, but i dont understand where it comes from
    # much like some of the parameters in the hough function

    for line in lines:
        # 1
        for x1, y1, x2, y2 in line:
            slope = get_slope(x1, y1, x2, y2)
            if slope > det_slope:
                r_slope.append(slope)
                r_lane.append(line)
            elif slope < -det_slope:
                l_slope.append(slope)
                l_lane.append(line)
        # 2
        y_global_min = min(y1, y2, y_global_min)

    # to prevent errors in challenge video from dividing by zero
    if (len(l_lane) == 0) or (len(r_lane) == 0):
        print('no lane detected')
        return 1

    # 3
    l_slope_mean = np.mean(l_slope, axis=0)
    r_slope_mean = np.mean(r_slope, axis=0)
    l_mean = np.mean(np.array(l_lane), axis=0)
    r_mean = np.mean(np.array(r_lane), axis=0)

    if (r_slope_mean == 0) or (l_slope_mean == 0):
        print('dividing by zero')
        return 1

    # 4, y=mx+b -> b = y -mx
    l_b = l_mean[0][1] - (l_slope_mean * l_mean[0][0])
    r_b = r_mean[0][1] - (r_slope_mean * r_mean[0][0])

    # 5, using y-extrema (#2), b intercept (#4), and slope (#3) solve for x using y=mx+b
    # x = (y-b)/m
    # these 4 points are our two lines that we will pass to the draw function
    l_x1 = int((y_global_min - l_b) / l_slope_mean)
    l_x2 = int((y_max - l_b) / l_slope_mean)
    r_x1 = int((y_global_min - r_b) / r_slope_mean)
    r_x2 = int((y_max - r_b) / r_slope_mean)

    # 6
    if l_x1 > r_x1:
        l_x1 = int((l_x1 + r_x1) / 2)
        r_x1 = l_x1
        l_y1 = int((l_slope_mean * l_x1) + l_b)
        r_y1 = int((r_slope_mean * r_x1) + r_b)
        l_y2 = int((l_slope_mean * l_x2) + l_b)
        r_y2 = int((r_slope_mean * r_x2) + r_b)
    else:
        l_y1 = y_global_min
        l_y2 = y_max
        r_y1 = y_global_min
        r_y2 = y_max

    current_frame = np.array([l_x1, l_y1, l_x2, l_y2, r_x1, r_y1, r_x2, r_y2], dtype="float32")

    if first_frame == 1:
        next_frame = current_frame
        first_frame = 0
    else:
        prev_frame = cache
        next_frame = (1 - alpha) * prev_frame + alpha * current_frame

    cv2.line(img, (int(next_frame[0]), int(next_frame[1])), (int(next_frame[2]), int(next_frame[3])), color, thickness)
    cv2.line(img, (int(next_frame[4]), int(next_frame[5])), (int(next_frame[6]), int(next_frame[7])), color, thickness)

    cache = next_frame


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img


def weighted_img(img, initial_img, alpha=0.8, beta=1., lamda=0.):
    """
    Put the lines on the original image.
    NOTE: initial_img and img must be the same shape!
    :param img: is the output of the hough_lines(), An image with lines drawn on it.
    It Should be a blank image (all black) with lines drawn on it.
    :param initial_img: Image before any processing.
    :param alpha:
    :param beta:
    :param lamda:
    :return: initial_img * α + img * β + λ
    """
    return cv2.addWeighted(initial_img, alpha, img, beta, lamda)


def main(img_dir):
    img = cv2.imread(img_dir)
    cv2.namedWindow("Image")  # Create a window to display image

    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # RGB 2 GRAY
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)  # Extract HSV

    # MASK
    lower_yellow = np.array([20, 100, 100], dtype="uint8")
    upper_yellow = np.array([30, 255, 255], dtype="uint8")
    mask_yellow = cv2.inRange(img_hsv, lower_yellow, upper_yellow)
    mask_white = cv2.inRange(img_gray, 200, 255)
    mask_yw = cv2.bitwise_or(mask_white, mask_yellow)
    mask_yw_image = cv2.bitwise_and(img_gray, mask_yw)

    # Gaussian Blur
    kernel_size = 5
    img_blurred = cv2.GaussianBlur(img_gray, (kernel_size, kernel_size), 0)

    # Canny Edge Detection
    low_threshold = 50
    high_threshold = 150
    img_canny = cv2.Canny(img_blurred, low_threshold, high_threshold)

    # Add ROI Mask
    img_shape = img.shape
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

    line_image = hough_lines(roi_image, rho, theta, threshold, min_line_len, max_line_gap)
    result = weighted_img(line_image, img, alpha=0.8, beta=1., lamda=0.)

    # Show Image in IDLE
    cv2.imshow("Image", mask_yw_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main(img_dir="./Images/lane1.jpeg")
