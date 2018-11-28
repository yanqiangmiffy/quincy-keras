# !/usr/bin/env python  
# -*- coding:utf-8 _*-  
""" 
@Author:yanqiang 
@File: 02_logistic_regression.py 
@Time: 2018/11/28 14:08
@Software: PyCharm 
@Description: 逻辑回归 手写字体识别
"""

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras.datasets import mnist
from keras.utils import np_utils

batch_size = 128
n_classes = 10  # 10 类
n_epoch = 100

# 加载MNIST数据集
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# print(X_train.shape,y_train.shape) # (60000, 28, 28) (60000,)
# print(X_test.shape,y_test.shape) # (10000, 28, 28) (10000,)
X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
y_train = np_utils.to_categorical(y_train, n_classes)
y_test = np_utils.to_categorical(y_test, n_classes)

# 逻辑回归模型
model = Sequential()
model.add(Dense(output_dim=10, input_shape=(784,), init='normal', activation='softmax'))
model.compile(optimizer=SGD(lr=0.05), loss='categorical_crossentropy', metrics=['accuracy'])

# 训练模型
history = model.fit(X_train, y_train, nb_epoch=n_epoch, batch_size=batch_size, verbose=1)

# 评价模型
evaluation = model.evaluate(X_test, y_test, verbose=1)
print('Summary: Loss over the test dataset: %.2f, Accuracy: %.2f' % (evaluation[0], evaluation[1]))
# Summary: Loss over the test dataset: 0.27, Accuracy: 0.92
