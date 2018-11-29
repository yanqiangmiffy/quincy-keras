# !/usr/bin/env python  
# -*- coding:utf-8 _*-  
""" 
@Author:yanqiang 
@File: 08_tensorboard.py 
@Time: 2018/11/29 14:00
@Software: PyCharm 
@Description:
"""
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.optimizers import RMSprop
from keras.datasets import mnist
from keras import initializers
from keras.utils import np_utils
from keras.callbacks import TensorBoard

# 参数设置
batch_size=128
n_epoch=20
n_classes=10
prob_drop_input=0.2
prob_drop_hidden=0.2


# 加载数据
(X_train,y_train),(X_test,y_test)=mnist.load_data()
X_train=X_train.astype('float32')/255.
X_test=X_test.astype('float32')/255.
X_train=X_train.reshape(60000,784)
X_test=X_test.reshape(10000,784)
y_train=np_utils.to_categorical(y_train,num_classes=n_classes)
y_test=np_utils.to_categorical(y_test,num_classes=n_classes)


# 构建模型
model=Sequential()
model.add(Dense(input_dim=784,output_dim=625,kernel_initializer='random_normal',activation='sigmoid',name='dense1'))
model.add(Dropout(prob_drop_input,name='dropout1'))
model.add(Dense(input_dim=625,output_dim=625,kernel_initializer='random_normal',activation='sigmoid',name='dense2'))
model.add(Dropout(prob_drop_hidden,name="dropout2"))
model.add(Dense(input_dim=625,output_dim=10,kernel_initializer='random_normal',activation='softmax',name='dense3'))
model.compile(optimizer=RMSprop(lr=0.001,rho=0.9),loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()

# 训练模型
tb_callback=TensorBoard(log_dir='./logs/09_tensorboard',
                        histogram_freq=1)
history=model.fit(X_train,y_train,
                  batch_size=batch_size,
                  epochs=n_epoch,
                  verbose=1,
                  validation_data=[X_test,y_test],
                  callbacks=[tb_callback])
# 评价模型
evaluation=model.evaluate(X_test,y_test,verbose=1)
print('Summary: Loss over the test dataset: %.2f, Accuracy: %.2f' % (evaluation[0], evaluation[1]))