# 加载系统工具库
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets

# 加载自定义的工具库
from init_utils import *

plt.rcParams['figure.figsize'] = (7.0, 4.0)  # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# 加载我们用算法生成的假数据并把它们画出来（只画了训练数据，没有画测试数据）。
# 我们的目的就是训练一个模型，使其能够将红点和蓝点区分开。
train_X, train_Y, test_X, test_Y = load_dataset()


# 构建一个模型，实现细节很多都在我们自定义的工具库init_utils.py里面。因为那些细节我们之前已经学过，
# 所以为了突出重点，就把它们隐藏在工具库里面了。
# 这个模型的特点是，它可以指定3种不同的初始化方法，通过参数initialization来控制
def model(X, Y, learning_rate=0.01, num_iterations=15000, print_cost=True, initialization="he"):
    grads = {}
    costs = []
    m = X.shape[1]
    layers_dims = [X.shape[0], 10, 5, 1]  # 构建一个3层的神经网络

    # 3种不同的初始化方法，后面会对这3种初始化方法进行详细介绍
    if initialization == "zeros":
        parameters = initialize_parameters_zeros(layers_dims)
    elif initialization == "random":
        parameters = initialize_parameters_random(layers_dims)
    elif initialization == "he":
        parameters = initialize_parameters_he(layers_dims)

    # 梯度下降训练参数
    for i in range(0, num_iterations):
        a3, cache = forward_propagation(X, parameters)
        cost = compute_loss(a3, Y)
        grads = backward_propagation(X, Y, cache)
        parameters = update_parameters(parameters, grads, learning_rate)

        if print_cost and i % 1000 == 0:
            print("Cost after iteration {}: {}".format(i, cost))
            costs.append(cost)

    # 画出成本走向图
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    return parameters


# 这是我们要介绍的第一种方法。是最差的方法。也是我们学习过的第一种方法——全部初始化为0
def initialize_parameters_zeros(layers_dims):
    parameters = {}
    L = len(layers_dims)

    for l in range(1, L):
        parameters['W' + str(l)] = np.zeros((layers_dims[l], layers_dims[l - 1]))
        parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))
    return parameters


def initialize_parameters_random(layers_dims):
    np.random.seed(3)
    parameters = {}
    L = len(layers_dims)

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l - 1]) * 10
        parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))
    return parameters


def  initialize_parameters_he(layers_dims):
    np.random.seed(3)
    parameters = {}
    L = len(layers_dims) - 1

    for l in range(1, L + 1):
        parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l - 1]) * np.sqrt(2 / layers_dims[l - 1])
        parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))

    return parameters


####TEST####

def initialize_parameters_zeros_test():
    # 单元测试
    parameters = initialize_parameters_zeros([3, 2, 1])
    print("W1 = " + str(parameters["W1"]))
    print("b1 = " + str(parameters["b1"]))
    print("W2 = " + str(parameters["W2"]))
    print("b2 = " + str(parameters["b2"]))


def predictions_test_zeros_test():
    # 用全0初始化法进行参数训练
    parameters = model(train_X, train_Y, initialization="zeros")
    print("On the train set:")
    predictions_train = predict(train_X, train_Y, parameters)  # 对训练数据进行预测，并打印出准确度
    print("On the test set:")
    predictions_test = predict(test_X, test_Y, parameters)  # 对训测试数据进行预测，并打印出准确度
    print("predictions_train = " + str(predictions_train))
    print("predictions_test = " + str(predictions_test))


def initialize_parameters_random_test():
    parameters = initialize_parameters_random([3, 2, 1])
    print("W1 = " + str(parameters["W1"]))
    print("b1 = " + str(parameters["b1"]))
    print("W2 = " + str(parameters["W2"]))
    print("b2 = " + str(parameters["b2"]))
    parameters = model(train_X, train_Y, initialization="random")
    print("On the train set:")
    predictions_train = predict(train_X, train_Y, parameters)
    print("On the test set:")
    predictions_test = predict(test_X, test_Y, parameters)
    print(predictions_train)
    print(predictions_test)

    plt.title("Model with large random initialization")
    axes = plt.gca()
    axes.set_xlim([-1.5, 1.5])
    axes.set_ylim([-1.5, 1.5])
    plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)


# 但是我们还可以继续参数的初始化方法，使神经网络更加强大。
#
# 从上面的成本图可以看出，成本开始时特别大。这是因为我们将参数初始化成了很大的值，这就会导致神经网络在前期对预测太绝对了，不是0就是1，如果预测错了，就会导致成本很大。
#
# 参数初始化得不对会导致训练效率很差，需要训练很长时间才能靠近理想值。下面的代码中，你可以将训练次数改大一些，你会看到，训练得越久，成本会越来越小，预测精准度越来越高。
#
# 参数初始化得不对，还会导致梯度消失和爆炸。
# 给大家演示一下我们文章2.1.8中提到的参数初始化方法。
def initialize_parameters_he_test():
    parameters = initialize_parameters_he([2, 4, 1])
    print("W1 = " + str(parameters["W1"]))
    print("b1 = " + str(parameters["b1"]))
    print("W2 = " + str(parameters["W2"]))
    print("b2 = " + str(parameters["b2"]))

    parameters = model(train_X, train_Y, initialization="he")
    print("On the train set:")
    predictions_train = predict(train_X, train_Y, parameters)
    print("On the test set:")
    predictions_test = predict(test_X, test_Y, parameters)
    print(predictions_train)
    print(predictions_test)
    plt.title("Model with He initialization")
    axes = plt.gca()
    axes.set_xlim([-1.5, 1.5])
    axes.set_ylim([-1.5, 1.5])
    plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)


if __name__ == '__main__':
    ### 初始化为0
    # initialize_parameters_zeros_test();
    # predictions_test_zeros_test()
    initialize_parameters_he_test()
