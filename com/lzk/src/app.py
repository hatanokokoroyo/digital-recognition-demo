# coding=utf-8
import sys

import numpy

import mnist_util
from neural_network import NeuralNetwork

network: NeuralNetwork


def init_neural_network():
    global network
    # 1.创建神经网络
    # 输入节点数: 28 * 28的像素点输入
    input_nodes = 784
    # 隐藏节点数: 一个适量的值
    hide_nodes = 128
    # 输出节点数: 对应要识别的10个数字
    output_nodes = 10
    learning_rate = 0.5
    network = NeuralNetwork(input_nodes, hide_nodes, output_nodes, learning_rate)
    # 2.开始训练
    images = mnist_util.load_simple_train_data()
    # images = mnist_util.load_train_data()
    total = len(images)
    cnt = 0
    for image in images:
        cnt += 1
        # 转换为浮点型输入数组
        inputs = numpy.asfarray(image.pixels) / 255.0 * 0.99 + 0.01
        # 10位数组, 正确结果对应位置的值为0.99, 其他为0.01
        targets = numpy.zeros(output_nodes) + 0.01
        targets[image.label] = 0.99
        network.train(inputs, targets)
        sys.stdout.write("训练进度: {0}/{1}\r".format(cnt, total))
    sys.stdout.flush()
    print("训练完成")


def test_neural_network():
    global network
    images = mnist_util.load_simple_test_data()
    # images = mnist_util.load_test_data()
    total = len(images)
    right_cnt = 0
    error_images = []
    for image in images:
        inputs = numpy.asfarray(image.pixels) / 255.0 * 0.99 + 0.01
        result = network.query(inputs)
        # print(str.format("期望结果: {0}, 识别结果: {1}", image.label, result))
        if result == image.label:
            right_cnt += 1
        else:
            image.result = result
            error_images.append(image)
    print("识别率为" + str(float(right_cnt) / float(total)) + ", 以下为识别错误的数字")
    for image in error_images:
        print(str.format("期望结果: {0}, 错误结果: {1}", image.label, image.result))


if __name__ == '__main__':
    init_neural_network()
    test_neural_network()
