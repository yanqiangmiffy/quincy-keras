## CNN 卷积神经网络

### 1 卷积
![](http://adventuresinmachinelearning.com/wp-content/uploads/2017/04/Moving-filter.jpg)
```text
out1=0.5in1+0.5in2+0.5in6+0.5in7
    =0.5×2.0+0.5×3.0+0.5×2.0+0.5×1.5
    =4.25
    
out2=0.5in2+0.5in3+0.5in7+0.5in8
    =0.5×3.0+0.5×0.0+0.5×1.5+0.5×0.5
    =2.5

```
### 2 池化
![](http://adventuresinmachinelearning.com/wp-content/uploads/2017/04/Max-pooling.jpg)
```text
out1=max(in1,in2,in6,in7)
out2=max(in3,in4,in8,in9)
out3=max(in5,pad1,in10,pad2)
```

### 3 实例
![](http://adventuresinmachinelearning.com/wp-content/uploads/2017/04/CNN-example-block-diagram.jpg)
```text
model = Sequential()
model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
                 activation='relu',
                 input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(64, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(1000, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))
```
### 4 参考
- [Convolutional Neural Networks Tutorial in TensorFlow](http://adventuresinmachinelearning.com/convolutional-neural-networks-tutorial-tensorflow/)
- [Keras tutorial – build a convolutional neural network in 11 lines](http://adventuresinmachinelearning.com/keras-tutorial-cnn-11-lines/)
- [adventures-in-ml-code](https://github.com/adventuresinML/adventures-in-ml-code)