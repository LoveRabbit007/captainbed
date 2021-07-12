import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
from tf_utils import load_dataset, random_mini_batches, convert_to_one_hot, predict

np.random.seed(1)


def linear_function():
    np.random.seed(1)

    X = tf.constant(np.random.randn(3, 1), name="X")  # 定义一个维度是(3, 1)的常量，randn函数会生成随机数
    W = tf.constant(np.random.randn(4, 3), name="W")
    b = tf.constant(np.random.randn(4, 1), name="b")
    Y = tf.add(tf.matmul(W, X), b)  # tf.matmul函数会执行矩阵运算

    # 创建session，然后用run来执行上面定义的操作
    sess = tf.Session()
    result = sess.run(Y)
    sess.close()
    print("result = " + str(linear_function()))
    return result


def sigmoid(z):
    x = tf.placeholder(tf.float32, name="x")  # 定义一个类型为float32的占位符

    sigmoid = tf.sigmoid(x)  # 调用tensorflow的sigmoid函数，并且将占位符作为参数传递进去

    with tf.Session() as sess:  # 创建一个session
        # 用run来执行上面定义的sigmoid操作。
        # 执行时将外面传入的z填充到占位符x中，也就相当于把z作为参数传入了tensorflow的sigmoid函数中了。
        result = sess.run(sigmoid, feed_dict={x: z})

    return result


def costs(z_in, y_in):
    z = tf.placeholder(tf.float32, name="z")  # 创建占位符
    y = tf.placeholder(tf.float32, name="y")

    # 使用sigmoid_cross_entropy_with_logits来构建cost操作。
    cost = tf.nn.sigmoid_cross_entropy_with_logits(logits=z, labels=y)

    # 创建session
    sess = tf.Session()

    # 将传入的z_in和y_in填充到占位符中，然后执行cost操作
    cost = sess.run(cost, feed_dict={z: z_in, y: y_in})

    sess.close()

    return cost


def print_sigmoid():
    print("sigmoid(0) = " + str(sigmoid(0)))
    print("sigmoid(12) = " + str(sigmoid(12)))


def cost_print():
    logits = np.array([0.2, 0.4, 0.7, 0.9])
    cost = costs(logits, np.array([0, 0, 1, 1]))
    print("cost = " + str(cost))


def one_hot_matrix(labels, C_in):
    """
    labels就是真实标签y向量；
    C_in就是类别的数量
    """

    # 创建一个名为C的tensorflow常量，把它的值设为C_in
    C = tf.constant(C_in, name='C')

    # 使用one_hot函数构建转换操作，将这个操作命名为one_hot_matrix。
    one_hot_matrix = tf.one_hot(indices=labels, depth=C, axis=0)

    sess = tf.Session()

    # 执行one_hot_matrix操作
    one_hot = sess.run(one_hot_matrix)

    sess.close()

    return one_hot


def one_hot_matrix_print():
    labels = np.array([1, 2, 3, 0, 2, 1])
    one_hot = one_hot_matrix(labels, C_in=4)
    print("one_hot = " + str(one_hot))


def ones(shape):
    # 将维度信息传入tf.ones中
    ones = tf.ones(shape)

    sess = tf.Session()

    # 执行ones操作
    ones = sess.run(ones)

    sess.close()

    return ones


def ones_print():
    print("ones = " + str(ones([3])))


if __name__ == '__main__':
    # print_sigmoid()
    # cost_print()
    one_hot_matrix_print()
    ones_print()
