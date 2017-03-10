# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 11:19:03 2017

@author: USUARIO
"""

import tensorflow as tf
import numpy as np

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

import pandas as pd
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)
#将verbosity（冗长度）设置为tf.logging.INFO以查看更详细的日志。

COLUMNS = ["crim", "zn", "indus", "nox", "rm", "age",
           "dis", "tax", "ptratio", "medv"]
FEATURES = ["crim", "zn", "indus", "nox", "rm",
            "age", "dis", "tax", "ptratio"]
LABEL = "medv"
#定义列名，为了区分特征变量和标签，还特意将两者分来，分别用FEATURES和LABEL表示


training_set = pd.read_csv("boston_train.csv", skipinitialspace=True,
                           skiprows=1, names=COLUMNS)
test_set = pd.read_csv("boston_test.csv", skipinitialspace=True,
                       skiprows=1, names=COLUMNS)
prediction_set = pd.read_csv("boston_predict.csv", skipinitialspace=True,
                             skiprows=1, names=COLUMNS)
#读取数据到pandas中。
###############################
feature_cols = [tf.contrib.layers.real_valued_column(k)
                  for k in FEATURES]
#利用tf.contrib.layers.real_valued_column来创建FeatureColumn,该行用于指定训练输入数据的特征
regressor = tf.contrib.learn.DNNRegressor(feature_columns=feature_cols,
                                          hidden_units=[10, 10],
                                          model_dir="/tmp/boston_model")
#进行DNN训练，两个隐藏层，各有10个神经元。

def input_fn(data_set):
  feature_cols = {k: tf.constant(data_set[k].values)
                  for k in FEATURES}
  labels = tf.constant(data_set[LABEL].values)
  return feature_cols, labels
#为了将数据输入到regressor中，需要有数据输入函数。接受dataframe的输入，并输出feature和label的组合的tensor

regressor.fit(input_fn=lambda: input_fn(training_set), steps=5000)
#对模型进行数据拟合。数据输入为训练集。
ev = regressor.evaluate(input_fn=lambda: input_fn(test_set), steps=1)
#进行评估
loss_score = ev["loss"]
print("Loss: {0:f}".format(loss_score))
#获得其中的损失的数值并查看
y = regressor.predict(input_fn=lambda: input_fn(prediction_set))
predictions = list(itertools.islice(y, 6))
print ("Predictions: {}".format(str(predictions)))
#模型进行预测,slice()(seq, [start,] stop [, step])为返回迭代器，其中的项目来自　将seq，从start开始,到stop结束，以step步长切割后
