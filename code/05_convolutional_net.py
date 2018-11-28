# !/usr/bin/env python  
# -*- coding:utf-8 _*-  
""" 
@Author:yanqiang 
@File: 05_convolutional_net.py 
@Time: 2018/11/28 18:21
@Software: PyCharm 
@Description: 卷积神经网络
"""
from keras.models import Sequential
from keras.layers import Dense,Flatten,Convolution2D,MaxPooling2D,Dropout
from keras.optimizers import RMSprop
from keras.datasets import mnist
from keras.utils import np_utils
from keras import initializers
from keras import backend as K


# 参数设置
batch_size=128
n_epoch=100
n_classes=10

