# !/usr/bin/env python  
# -*- coding:utf-8 _*-  
""" 
@Author:yanqiang 
@File: 03_net.py.py 
@Time: 2018/11/28 16:11
@Software: PyCharm 
@Description: 前馈神经网络 多层感知机
"""
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras.datasets import mnist
from keras.utils import np_utils

# 参数
batch_size=128
n_classes=10
n_epoch=100

# 加载MNIST数据集
(X_train,y_train),(X_test,y_test)=mnist.load_data()
X_train=X_train.reshape(60000,784)
X_test=X_test.reshape(10000,784)
X_train=X_train.astype('float32')
X_test=X_test.astype('float32')
X_train/=255
X_test/=255
y_train=np_utils.to_categorical(y_train,num_classes=n_classes)
y_test=np_utils.to_categorical(y_test,num_classes=n_classes)

# 多层感知机模型
model=Sequential()
model.add(Dense(input_dim=784,output_dim=625,init='normal',activation='sigmoid'))
model.add(Dense(input_dim=625,output_dim=625,init='normal',activation='sigmoid'))
model.add(Dense(input_dim=625,output_dim=10,init='normal',activation='softmax'))
model.compile(optimizer=SGD(lr=0.05),loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()

# 训练模型
history=model.fit(X_train,y_train,batch_size=batch_size,epochs=n_epoch,verbose=1,validation_split=0.2)

# 模型评估

evaluation=model.evaluate(X_test,y_test,verbose=1)
print("Summary:Loss over the test dataset:%.2f,Accuracy %.2f" % (evaluation[0],evaluation[1]))
# Summary:Loss over the test dataset:0.12,Accuracy 0.96


