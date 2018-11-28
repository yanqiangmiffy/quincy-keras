# !/usr/bin/env python  
# -*- coding:utf-8 _*-  
""" 
@Author:yanqiang 
@File: 04_modern_net.py 
@Time: 2018/11/28 17:44
@Software: PyCharm 
@Description: Dropout
"""
from keras.models import Sequential
from keras.layers import Dense,Activation,Dropout
from keras.optimizers import RMSprop
from keras.datasets import mnist
from keras.utils import np_utils

# 参数设置
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


# 深层多层感知机
model=Sequential()
model.add(Dense(input_dim=784,units=625,kernel_initializer='normal'))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(input_dim=625,units=625,kernel_initializer='normal'))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(input_dim=625,units=n_classes,kernel_initializer='normal'))
model.add(Activation('softmax'))

model.compile(optimizer=RMSprop(lr=0.001,rho=0.9),loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()

# 训练
history=model.fit(X_train,y_train,epochs=n_epoch,batch_size=batch_size,verbose=1)

# 评估模型
evaluation=model.evaluate(X_test,y_test,verbose=1)
print("Summary：Loss over the test dataset:%.2f,Accuracy %.2f" % (evaluation[0],evaluation[1]))
# Summary：Loss over the test dataset:0.19,Accuracy 0.98
