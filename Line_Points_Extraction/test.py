# -*- coding: utf-8 -*-
# @Time    : 2020/3/12 3:57 下午
# @Author  : Yingke Ding
# @FileName: test.py
# @Software: PyCharm

import numpy as np
import cv2


def access_pixels(img):
    """遍历图像每个像素的每个通道"""
    print(img.shape)  # 打印图像的高，宽，通道数（返回一个3元素的tuple）
    height = img.shape[0]  # 将tuple中的元素取出，赋值给height，width，channels
    width = img.shape[1]
    channels = img.shape[2]
    print("height:%s,width:%s,channels:%s" % (height, width, channels))
    print(img.size)  # 打印图像数组内总的元素数目（总数=高X宽X通道数）
    for row in range(height):  # 遍历每一行
        for col in range(width):  # 遍历每一列
            for channel in range(channels):  # 遍历每个通道（三个通道分别是BGR）
                img[row][col][channel] = 255 - img[row][col][channel]
                # 通过数组索引访问该元素，并作出处理
    cv2.imshow("processed img", img)  # 将处理后的图像显示出来


# 上述自定义函数的功能是像素取反，当然，opencv自带像素取反方法bitwise_not()，不需要这么麻烦
def inverse(img):
    """
    此函数与access_pixels函数功能一样
    :param img:
    :return:
    """

    dst = cv2.bitwise_not(img)
    cv2.imshow("inversed_img", dst)


def create_img():
    """#创建一张三通道图像"""
    img = np.zeros([600, 800, 3], dtype=np.uint8)
    # 创建高600像素，宽800像素，每个像素有BGR三通道的数组（图像）
    # 由于元素都在0~255之内，规定数组元素类型为uint8已足够
    img[:, :, 2] = np.ones([600, 800]) * 255
    # img[:,:,2]是一种切片方式，冒号表示该维度从头到尾全部切片取出
    # 所以img[:,:,2]表示切片取出所有行，所有列的第三个通道（索引为2）
    # 右侧首先创建了一个600X800的二维数组，所有元素初始化为1，再乘上255，即所有元素变为255
    # 注意右侧600X800的二维数组与左侧切片部分形状相同，所以可以赋值
    # 即所有行，所有列的第三个通道(R)的值都变为255，一二通道(BG)仍为0，即所有像素变为红色BGR(0，0，255)
    cv2.imshow("created_img", img)


def create_img_1():
    """创建一张单通道图像"""
    img = np.zeros([400, 400, 1], dtype=np.uint8)
    # 高400像素，宽400像素，单通道
    # 仍是三维数组，不过第三个维度长度为1，用来表示像素的灰度（0~255）
    img[:, :, 0] = np.ones([400, 400]) * 127
    # 切片取出所有行所有列的第一个元素（索引为0），灰度元素，并赋值为127
    cv2.imshow("created_img1", img)


src = cv2.imread("Z.PNG")  # 读取图像
t1 = cv2.getTickCount()  # 记录下起始时刻
access_pixels(src)  # 访问图像的每个元素并处理
create_img()  # 通过numpy创建三通道彩色图
create_img_1()  # 通过numpy创建单通道灰度图
t2 = cv2.getTickCount()  # 记录下结束时刻
time = ((t2 - t1) * 1000) / cv2.getTickFrequency()  # 计算中间语句执行所耗时间，单位为ms
print("所耗时间：%s" % time)

cv2.waitKey(0)
cv2.destroyAllWindows()
