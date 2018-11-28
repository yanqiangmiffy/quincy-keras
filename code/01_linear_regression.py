# !/usr/bin/env python  
# -*- coding:utf-8 _*-  
""" 
@Author:yanqiang 
@File: 01_linear_regression.py
@Time: 2018/11/28 11:30
@Software: PyCharm 
@Description: 线性回归
"""
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
import numpy as np
from keras import losses

# 生成数据集
X_train=np.linspace(-1,1,101)
# ＃创建一个近似线性但具有一些随机噪声的y值
y_train=2*X_train+np.random.randn(*X_train.shape)*0.33
print(X_train[:10])
print(y_train[:10])
# 线性回归模型
model=Sequential()
model.add(Dense(output_dim=1,input_dim=1,init='normal',activation='linear'))
model.compile(optimizer=SGD(lr=0.01),loss='mean_squared_error')
# model.compile(optimizer=SGD(lr=0.01),loss=losses.mean_absolute_error) # loss也可以这样写
# model.compile(optimizer='sgd',loss='mse') # loss也可以这样写

# 训练
model.fit(X_train,y_train,nb_epoch=100,verbose=1)

