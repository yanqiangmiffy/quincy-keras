## tensorboard
在命令行下运行：
```text
tensorboard --logdir ./logs 
```
然后访问：http://localhost:6006

![](https://upload-images.jianshu.io/upload_images/1531909-83f6f2e3ce846110.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

## History对象

History对象在keras callbacks.py文件里

```
class History(Callback):
    """Callback that records events into a `History` object.

    This callback is automatically applied to
    every Keras model. The `History` object
    gets returned by the `fit` method of models.
    """

    def on_train_begin(self, logs=None):
        self.epoch = []
        self.history = {}

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epoch.append(epoch)
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)
```

可以看出History类对象包含两个属性，分别为epoch和history，epoch为训练轮数，history字典类型，包含val_loss,val_acc,loss,acc四个key值。