# coding=utf-8
import sys

import numpy as np

simple_train_file_path = "resources/mnist_train_100.csv"
simple_test_file_path = "resources/mnist_test_10.csv"
train_file_path = "resources/mnist_train.csv"
test_file_path = "resources/mnist_test.csv"


class ImageData:
    def __init__(self, label: int, pixels: []):
        self.label = label
        self.pixels = pixels


def load_simple_train_data():
    return load_data(simple_train_file_path)


def load_simple_test_data():
    return load_data(simple_test_file_path)


def load_train_data():
    return load_data(train_file_path)


def load_test_data():
    return load_data(test_file_path)


def load_data(file_path: str):
    """读取文件, 拆分数据, -> (data, label)"""
    with open(file_path, "r", encoding='utf-8') as r:
        lines = r.readlines()
    images = []
    for line in lines:
        image = _convert_data(line)
        images.append(image)
    return images


def _convert_data(line):
    """转换数据, 把原数据拆分为一个数字和一个数组 -> label, img"""
    data = line.split(',')
    # 转为整形数组
    data = list(map(int, data))
    # 图像对应的数字
    label = data[0]
    # 图像对应的像素点, 范围是0-255
    pixels = data[1:]
    return ImageData(label, pixels)


if __name__ == '__main__':
    load_train_data()
