import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import math
import sklearn
import sklearn.datasets

from opt_utils import load_params_and_grads, initialize_parameters, forward_propagation, backward_propagation
from opt_utils import compute_cost, predict, predict_dec, plot_decision_boundary, load_dataset
from testCases import *

plt.rcParams['figure.figsize'] = (7.0, 4.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


# 初始化指数加权平均值字典

def initialize_velocity(parameters):
    L = len(parameters) // 2  # 获取神经网络的层数
    v = {}

    # 循环每一层
    for l in range(L):
        # 因为l是从0开始的，所以下面要在l后面加上1
        # zeros_like会返回一个与输入参数维度相同的数组，而且将这个数组全部设置为0
        # 指数加权平均值字典的维度应该是与梯度字典一样的，而梯度字典是与参数字典一样的，所以zeros_like的输入参数是参数字典
        v["dW" + str(l + 1)] = np.zeros_like(parameters["W" + str(l + 1)])
        v["db" + str(l + 1)] = np.zeros_like(parameters["b" + str(l + 1)])

    return v


def initialize_velocity_test_case_print():
    parameters = initialize_velocity_test_case()

    v = initialize_velocity(parameters)
    print("v[\"dW1\"] = " + str(v["dW1"]))
    print("v[\"db1\"] = " + str(v["db1"]))
    print("v[\"dW2\"] = " + str(v["dW2"]))
    print("v[\"db2\"] = " + str(v["db2"]))


# 使用动量梯度下降算法来更新参数

def update_parameters_with_momentum(parameters, grads, v, beta, learning_rate):
    L = len(parameters) // 2

    # 遍历每一层
    for l in range(L):
        # 算出指数加权平均值。
        # 下面的beta就相当于我们文章中的k。
        # 看这段代码时应该回想一下我们文章中学到的“一行代码搞定指数加权平均值”的知识点
        v["dW" + str(l + 1)] = beta * v["dW" + str(l + 1)] + (1 - beta) * grads['dW' + str(l + 1)]
        v["db" + str(l + 1)] = beta * v["db" + str(l + 1)] + (1 - beta) * grads['db' + str(l + 1)]

        # 用指数加权平均值来更新参数
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * v["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * v["db" + str(l + 1)]

    return parameters, v


def update_parameters_with_momentum_test_case_print():
    parameters, grads, v = update_parameters_with_momentum_test_case()

    parameters, v = update_parameters_with_momentum(parameters, grads, v, beta=0.9, learning_rate=0.01)
    print("W1 = " + str(parameters["W1"]))
    print("b1 = " + str(parameters["b1"]))
    print("W2 = " + str(parameters["W2"]))
    print("b2 = " + str(parameters["b2"]))
    print("v[\"dW1\"] = " + str(v["dW1"]))
    print("v[\"db1\"] = " + str(v["db1"]))
    print("v[\"dW2\"] = " + str(v["dW2"]))
    print("v[\"db2\"] = " + str(v["db2"]))


if __name__ == '__main__':
    initialize_velocity_test_case_print()
    update_parameters_with_momentum_test_case_print()
