# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 14:44:49 2017

@author: USUARIO
"""

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

import tensorflow as tf
import numpy as np
sess = tf.InteractiveSession()
  
x = tf.placeholder(tf.float32, shape=[None, 784])
#定义x变量
y_ = tf.placeholder(tf.float32, shape=[None, 10])

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)
#建立函数weighted_variable,tf.truncated_normal从截尾的高斯分布输出随机的初始化参数数值。
#shape参数设定输出tensor的形状。然后将initial作为变量输出

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)
#建立函数bias_variable， 将bias设置为1的常数。

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
#利用tf.nn模块下的vonv2d函数，给定了4-D输入，过滤器的tensor然后计算2-D的卷积。
#x为输入，W是过滤器，strides设定步长，padding设定zero-padding.
#padding可以设置为‘SAME’或者‘VALID’两种类型，这里选'SAME'-使得卷积层和输入层的size一样。
#conv2d的输出也是一个tensor

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
#tf.nn下的max_pool是设置max pooling的函数。x是一个4-D的输入。

###第一个卷积层###
W_conv1 = weight_variable([5, 5, 1, 32])
#调用weight_variable，生成指定维度的tensor。该tensor就是过滤器。
b_conv1 = bias_variable([32])
#生成bias
x_image = tf.reshape(x, [-1,28,28,1])
#将输入变形，28*28像素，通道为1.
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
#先用conv2d函数，对变形后的图像进行卷积变换（过滤器为W_conv1),加上偏差后，进行relu变换
h_pool1 = max_pool_2x2(h_conv1)
#对上面的结果进行maxｐｏｏｌｉｎｇ

####第二层卷积层####
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
##由于一次使用了32个过滤器，所以输出是28*28*32的一个卷积层。这里在此基础上，使用64个5*5的过滤器。
#然后生成指定长度的值全为0.1的tensor作为bias
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)
#继续进行卷积变换和pooling

######densely connected layers#####
###经过两次max pooling，图像的大小已经被压缩到了7*7
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
##由于上一层是64个过滤器，那么图像维度成为7*7*64，然后本层添加1024个与上一层全连接的神经元。因而这一层的参数数量是7 * 7 * 64*1024
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
##将上一层pooling后的结果‘熨平’成7*7*64的向量，然后对其进行relu变换进行全连接。这里的权重就是上两行W_fc1和b_fc1设定的权重.


keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
#为了防止过度拟合，最后还有一个dropout的过程。最后全连接层的神经元有一定概率被舍弃掉。
##设定每隔神经元被保留的概率是keep_prob，然后使用nn.dropout实现dropout。
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
##最后，在最终输出层添加一岑，类似于进行一个线性变换。

cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
#nn.softmax_cross_entropy_with_logits computes softmax cross entropy between logits and labels.
#labels: Each row labels[i] must be a valid probability distribution.
#logits: Unscaled log probabilities.

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
#使用Adam最优化方法，以进行交叉熵最小化为目标，进行最优化。

correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
#判断某个分类是否正确。如果预测类（tf.argmax(y_conv,1)）与实际值相等，那么取值为1，反之为0.
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#accuracy tensor：为平均正确率。（先使用tf.cast将前面的预测正确与否变成tf.float32的类型）
sess.run(tf.global_variables_initializer())
#初始化全局变量
for i in range(20000):
  batch = mnist.train.next_batch(50)
  if i%100 == 0:
    train_accuracy = accuracy.eval(feed_dict={
        x:batch[0], y_: batch[1], keep_prob: 1.0})
    print("step %d, training accuracy %g"%(i, train_accuracy))
  train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
#使用if循环，进行20000次迭代的adam SGD。每个batch的规模选择50。
#每迭代100次，输出以下内容：第i步，训练准确率为。。。其中训练准确是根据当前batch的数据以100%的保留率计算的。
#然后，对于0-20000的所有i，也就是对于每个batch，迭代进行adam optimizor的最优化，保留率为50%
print("test accuracy %g"%accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
#20000次迭代训练后，最终通过验证集进行准确率的验证。验证集可以在feed_dict中进行设置。保留率为100%。
