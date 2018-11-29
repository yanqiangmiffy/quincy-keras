## ModelCheckpoint
```text
keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
```
`filepath`可以是格式化的字符串，里面的占位符将会被`epoch`值和传入`on_epoch_end`的`logs`关键字所填入

例如，`filepath`若为`weights.{epoch:02d-{val_loss:.2f}}.hdf5`，则会生成对应`epoch`和验证集`loss`的多个文件。

## 模型恢复
```text
loaded_model = load_model('./logs/model_mlp')
loaded_model.load_weights('./logs/weights.epoch.09-val_loss.0.08.hdf5')
loaded_model.summary()
```
