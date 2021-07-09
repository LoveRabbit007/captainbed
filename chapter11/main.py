import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
from tf_utils import load_dataset, random_mini_batches, convert_to_one_hot, predict

np.random.seed(1)

# 定义一个tensorflow常量
y_hat = tf.constant(36, name='y_hat')
y = tf.constant(39, name='y')
# 定义一个tensorflow变量，这个变量就表示了上面的loss函数
loss = tf.Variable((y - y_hat) ** 2, name='loss')
# 这个可以看作是tensorflow的固定写法，后面会使用init来初始化loss变量
init = tf.global_variables_initializer()
# 创建一个tensorflow的session
with tf.Session() as session:
    # 用这个session来执行init初始化操作
    session.run(init)
    # 用session来执行loss操作，并将loss的值打印处理
    print(session.run(loss))


#  看到上面的代码，刚接触tensorflow的你可能会感觉到很多地方怪怪的，不是特别理解。这是正常的反应。每一种框架都有其自己的设计特色，当我们和它接触的次数越来越多后，就自然而然地理解那些特色了。所以现在不要求大家能理解它们，顺其自然先往下学就好，后面我保证你能自然而然地茅塞顿开！
#
# 编写tensorflow程序的一般步骤如下：
#
# 创建变量，在tensorflow里面有张量（Tensor）一词。
# 用这些张量来构建一些操作，例如上面的 (𝑦̂ (𝑖)−𝑦(𝑖))2
# 初始化那些张量。这个与我们平时的编程有一点不同，在tensorflow里面创建张量时，并没有对它们进行初始化。要到后面时用特定的语句来初始化那样张量。这样的设计正是tensorflow的一大亮点，它可以大大提升程序的运行效率。后面我们再详细解释它。
# 创建一个session。tensorflow里面用session来执行操作，前面只是定义了操作，必须要用session才能使那些操作被执行。
# 用session执行前面定义的操作。
# 上面的代码中，我们就创建了一个张量loss，但并没有计算它的值。要到后面执行了session.run(loss)语句后，才开始计算loss的值。给大家打个比方吧，session.run之前的都是在设想，session.run时才是执行那些设想。就像我们建一座大厦一样，session.run之前都是在设计，session.run时才是按设计图动工。
def pirnt_info():
    a = tf.constant(2)
    b = tf.constant(10)
    c = tf.multiply(a, b)
    print(c)
    sess = tf.Session()
    print(sess.run(c))


if __name__ == '__main__':
    pirnt_info()
