## keras评价函数的用法
评价函数用于评估当前训练模型的性能。当模型编译后（compile），评价函数应该作为 metrics 的参数来输入。

```text
model.compile(loss='mean_squared_error',
              optimizer='sgd',
              metrics=['mae', 'acc'])
```
```text
from keras import metrics
model.compile(loss='mean_squared_error',
              optimizer='sgd',
              metrics=[metrics.mae, metrics.categorical_accuracy])
```

## 可使用的评价函数
- binary_accuracy
`binary_accuracy(y_true, y_pred)`
- categorical_accuracy
`categorical_accuracy(y_true, y_pred)`
- sparse_categorical_accuracy
`sparse_categorical_accuracy(y_true, y_pred)`
- top_k_categorical_accuracy
`top_k_categorical_accuracy(y_true, y_pred, k=5)`
- sparse_top_k_categorical_accuracy
`sparse_top_k_categorical_accuracy(y_true, y_pred, k=5)`

## 自定义评价函数

自定义评价函数应该在编译的时候（compile）传递进去。该函数需要以 (y_true, y_pred) 作为输入参数，并返回一个张量作为输出结果。
```text
import keras.backend as K

def mean_pred(y_true, y_pred):
    return K.mean(y_pred)
    
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy', mean_pred])
```
