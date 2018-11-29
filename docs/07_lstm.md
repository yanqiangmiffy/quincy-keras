## LSTM 
> Long Short Term Memory

![](http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-var-GRU.png)

## pad_sequences
```text
keras.preprocessing.sequence.pad_sequences(sequences, maxlen=None, dtype='int32', padding='pre', truncating='pre', value=0.0)
```
将多个序列截断或补齐为相同长度。


**参数**

- sequences: 列表的列表，每一个元素是一个序列。
- maxlen: 整数，所有序列的最大长度。
- dtype: 输出序列的类型。
- padding: 字符串，'pre' 或 'post' ，表示长度不足时是在序列的前端补齐还是在后端补齐。
- truncating: 字符串，'pre' 或 'post' ，移除长度大于 maxlen 的序列的值，要么在序列前端截断，要么在后端。
- value: 浮点数，表示用来补齐的值。