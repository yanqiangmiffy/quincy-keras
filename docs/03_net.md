## 训练模型
```
history=model.fit(X_train,y_train,batch_size=batch_size,epochs=n_epoch,verbose=1,validation_split=0.2)
```

模型的fit函数有两个参数，shuffle用于将数据打乱，validation_split用于在没有提供验证集的时候，按一定比例从训练集中取出一部分作为验证集

这里有个陷阱是，程序是先执行validation_split，再执行shuffle的，所以会出现这种情况：

假如你的训练集是有序的，比方说正样本在前负样本在后，又设置了validation_split，那么你的验证集中很可能将全部是负样本

同样的，这个东西不会有任何错误报出来，因为Keras不可能知道你的数据有没有经过shuffle，保险起见如果你的数据是没shuffle过的，最好手动shuffle一下

## DENSE API更新

```text
# 多层感知机模型
model=Sequential()
model.add(Dense(input_dim=784,output_dim=625,kernel_initializer='normal',activation='sigmoid'))
model.add(Dense(input_dim=625,output_dim=625,kernel_initializer='normal',activation='sigmoid'))
model.add(Dense(input_dim=625,output_dim=10,kernel_initializer='normal',activation='softmax'))
model.compile(optimizer=SGD(lr=0.05),loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()
```
运行上面的代码会报如下警告：
```text
E:\ProgramData\Anaconda3\lib\site-packages\ipykernel_launcher.py:3: UserWarning: Update your `Dense` call to the Keras 2 API: `Dense(input_dim=784, kernel_initializer="normal", activation="sigmoid", units=625)`
  This is separate from the ipykernel package so we can avoid doing imports until
E:\ProgramData\Anaconda3\lib\site-packages\ipykernel_launcher.py:4: UserWarning: Update your `Dense` call to the Keras 2 API: `Dense(input_dim=625, kernel_initializer="normal", activation="sigmoid", units=625)`
  after removing the cwd from sys.path.
E:\ProgramData\Anaconda3\lib\site-packages\ipykernel_launcher.py:5: UserWarning: Update your `Dense` call to the Keras 2 API: `Dense(input_dim=625, kernel_initializer="normal", activation="softmax", units=10)`
  """
```

按照警告提示，代码修改为如下：
```text
# 多层感知机模型
model=Sequential()
model.add(Dense(input_dim=784,units=625,kernel_initializer='normal',activation='sigmoid'))
model.add(Dense(input_dim=625,units=625,kernel_initializer='normal',activation='sigmoid'))
model.add(Dense(input_dim=625,units=10,kernel_initializer='normal',activation='softmax'))
model.compile(optimizer=SGD(lr=0.05),loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()
```