# coding=utf-8
import numpy
import scipy.special


class NeuralNetwork:
    """支持三层结构的神经网络"""

    def __init__(self, input_nodes, hide_nodes, output_nodes, learning_rate):
        # 输入层节点数
        self.input_nodes = input_nodes
        # 隐藏层节点数
        self.hide_nodes = hide_nodes
        # 输出层节点数
        self.output_nodes = output_nodes
        # 学习率
        self.learning_rate = learning_rate
        # 输入层和隐层层的链接权重数组 input_nodes * hide_nodes
        self.wih = numpy.random.normal(0.0, pow(self.input_nodes, -0.5), (self.hide_nodes, self.input_nodes))
        # 隐层层和输出层的链接权重数组 hide_nodes * output_nodes
        self.who = numpy.random.normal(0.0, pow(self.hide_nodes, -0.5), (self.output_nodes, self.hide_nodes))
        # sigmoid函数
        self.s_function = lambda x: scipy.special.expit(x)

    def train(self, inputs_list, targets_list):
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T

        # 隐藏层输入 = 输入层和隐层层的链接权重 点乘 输入
        hidden_inputs = numpy.dot(self.wih, inputs)
        # 隐藏层输出结果
        hidden_outputs = self.s_function(hidden_inputs)
        # 隐藏层输出结果传输到输出层
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # 输出层的最终输出结果
        final_outputs = self.s_function(final_inputs)
        # 计算最终输出节点的误差(target - actual)
        output_errors = targets - final_outputs
        # 计算隐藏节点的误差
        hidden_errors = numpy.dot(self.who.T, output_errors)
        # 计算隐藏层和输出层之间新的权重
        self.who += self.learning_rate * numpy.dot(
            (output_errors * final_outputs * (1.0 - final_outputs)), numpy.transpose(hidden_outputs)
        )
        # 计算输入层和隐藏层之间新的权重
        self.wih += self.learning_rate * numpy.dot(
            (hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), numpy.transpose(inputs)
        )

    def query(self, inputs_list):
        outputs = self._calculate_final_outputs(inputs_list)
        max_point = 0
        for i in range(0, len(outputs) - 1):
            if outputs[max_point] < outputs[i]:
                max_point = i
        return max_point

    def _calculate_final_outputs(self, inputs_list):
        # 输入
        inputs = numpy.array(inputs_list, ndmin=2).T
        # 隐藏层输入
        hidden_inputs = numpy.dot(self.wih, inputs)
        # 隐藏层输出
        hidden_outputs = self.s_function(hidden_inputs)
        # 输出层输入
        final_input = numpy.dot(self.who, hidden_outputs)
        # 输出层输出
        final_output = self.s_function(final_input)
        return final_output
