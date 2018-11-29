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
from keras.layers import Dense,Flatten,Conv2D,MaxPooling2D,Dropout
from keras.optimizers import RMSprop
from keras.datasets import mnist
from keras.utils import np_utils
from keras import initializers
from keras import backend as K


# 参数设置
batch_size=128
n_epoch=10
n_classes=10

# 图片维度
img_rows,img_cols=28,28
# max pooling大小
pool_size=(2,2)

# dropout 比例
prob_drop_conv=0.2
prob_drop_hidden=0.5


# 加载MNIST数据集
(X_train,y_train),(X_test,y_test)=mnist.load_data()
print('X_train original shape:',X_train.shape)

if K.image_dim_ordering()=='th':
    # For theano backend
    X_train=X_train.reshape(X_train.shape[0],1,img_rows,img_cols)
    X_test=X_test.reshape(X_test.shape[0],1,img_rows,img_cols)
    input_shape=(1,img_rows,img_cols)
else:
    # For Tensorflow backend
    X_train=X_train.reshape(X_train.shape[0],img_rows,img_cols,1)
    X_test=X_test.reshape(X_test.shape[0],img_rows,img_cols,1)
    input_shape=(img_rows,img_cols,1)

X_train=X_train.astype('float32')/255.
X_test=X_test.astype('float32')/255.
y_train=np_utils.to_categorical(y_train,n_classes)
y_test=np_utils.to_categorical(y_test,n_classes)
print("X_train.shape:",X_train.shape)
print(X_train.shape[0],"train samples")
print(X_test.shape[0],"test samples")


# 卷积模型
model=Sequential()

# conv1 layer
model.add(Conv2D(filters=32,kernel_size=(3,3),padding='same',activation='relu',input_shape=input_shape))
model.add(MaxPooling2D(pool_size=pool_size,strides=(2,2),padding='same'))
model.add(Dropout(prob_drop_conv))

# conv2 layer
model.add(Conv2D(filters=64,kernel_size=(3,3),padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=pool_size,strides=(2,2),padding='same'))
model.add(Dropout(prob_drop_conv))

# conv3 layer
model.add(Conv2D(filters=128,kernel_size=(3,3),padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=pool_size,strides=(2,2),padding='same'))
model.add(Flatten())
model.add(Dropout(prob_drop_conv))

# fc1 layer
model.add(Dense(625,activation='relu',kernel_initializer='random_uniform'))
model.add(Dropout(prob_drop_hidden))

# fc2 layer
model.add(Dense(10,activation='softmax'))

opt=RMSprop(lr=0.01,rho=0.9)
model.compile(optimizer=opt, loss='categorical_crossentropy',metrics=['accuracy'])

# 训练
history=model.fit(X_train,y_train,epochs=n_epoch,batch_size=256,verbose=1)

# 评估模型
evaluation=model.evaluate(X_test,y_test,batch_size=256,verbose=1)
print("Summary: Loss over the test dataset:%.2f,Accuracy:%.2f" % (evaluation[0],evaluation[1]))
# Summary: Loss over the test dataset:0.10,Accuracy:0.98
