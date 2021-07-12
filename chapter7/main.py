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


def update_parameters_with_gd(parameters, grads, learning_rate):
    L = len(parameters) // 2  # 获取神经网络的层数。这里除以2是因为字典里面包含了w和b两种参数。

    # 遍历每一层
    for l in range(L):
        # 下面使用l + 1，是因为l是从0开始的，而我们的参数字典是从1开始的
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]

    return parameters


parameters, grads, learning_rate = update_parameters_with_gd_test_case()

parameters = update_parameters_with_gd(parameters, grads, learning_rate)


def update_parameters_with_gd_print():
    print("W1 = " + str(parameters["W1"]))
    print("b1 = " + str(parameters["b1"]))
    print("W2 = " + str(parameters["W2"]))
    print("b2 = " + str(parameters["b2"]))


def random_mini_batches(X, Y, mini_batch_size=64, seed=0):
    np.random.seed(seed)
    m = X.shape[1]  # 获取样本数量
    mini_batches = []

    # 第一步: 洗牌训练集
    permutation = list(np.random.permutation(m))  # 这行代码会生成m范围内的随机整数，如果m是3，那么结果可能为[2, 0, 1]
    shuffled_X = X[:, permutation]  # 这个代码会将X按permutation列表里面的随机索引进行洗牌。为什么前面是个冒号，因为前面是特征，后面才代表样本数
    shuffled_Y = Y[:, permutation].reshape((1, m))

    # 第二步: 分割洗牌后的训练集
    num_complete_minibatches = math.floor(m / mini_batch_size)  # 获取子训练集的个数（不包括后面不满mini_batch_size的那个子训练集）
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k * mini_batch_size:(k + 1) * mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size:(k + 1) * mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # 出来后面不满mini_batch_size的那个子训练集
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size:]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size:]

        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches


def random_mini_batches_print():
    X_assess, Y_assess, mini_batch_size = random_mini_batches_test_case()
    mini_batches = random_mini_batches(X_assess, Y_assess, mini_batch_size)
    print("第一个mini_batch_X的维度: " + str(mini_batches[0][0].shape))
    print("第二个mini_batch_X的维度: " + str(mini_batches[1][0].shape))
    print("第三个mini_batch_X的维度: " + str(mini_batches[2][0].shape))
    print("第一个mini_batch_Y的维度: " + str(mini_batches[0][1].shape))
    print("第二个mini_batch_Y的维度: " + str(mini_batches[1][1].shape))
    print("第三个mini_batch_Y的维度: " + str(mini_batches[2][1].shape))


if __name__ == '__main__':
    # update_parameters_with_gd_print()
    random_mini_batches_print()
