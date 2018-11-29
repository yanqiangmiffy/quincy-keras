## AutoEncoder
- 网络结构示意图1:

![](https://pic2.zhimg.com/80/v2-b2e5d5b40f7d58f1d1bf129d1f0c3a61_hd.jpg)

- 网络结构示意图2:

![](http://res.cloudinary.com/dyd911kmh/image/upload/f_auto,q_auto:best/v1522830223/Autoencoder_structure_af1jh8.png)

- 网络结构示意图3:

![](http://res.cloudinary.com/dyd911kmh/image/upload/f_auto,q_auto:best/v1522830223/AutoEncoder_kfqad1.png)

- 网络结果示意图4:

![](https://img-blog.csdn.net/20170331190710252?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvbWFyc2poYW8=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)

## Autoencoders的多种变体
- Undercomplete Autoencoders
- Regularized Autoencoders
- Sparse Autoencoders
- Denoising Autoencoders
- Contractive Autoencoder
- Stacked Autoencoders

## Denoising AutoEncoder
> Denoising AutoEncoder（DAE）是在“Vincent Extracting and composing robust features with denoising autoencoders, 2008”中提出的。本质就是在原样本中增加噪声，并期望利用 DAE 将加噪样本来还原成纯净样本。

> 本质上是说，为了让模型学到真正有用的表达向量，把噪声注入进训练样本，让模型还原注入噪声前的样本情况。这样对应模型就有了鉴别噪声，学到样本真正信息的能力，不会学到一个简单的identity函数




详情请看[深度学习基础：Autoencoders](https://zhuanlan.zhihu.com/p/34201555)
## AutoEncoder 作用

![](https://pic2.zhimg.com/80/v2-948d5ede26f9e6c0e484640b92c1dcad_hd.png)

原来有时神经网络要接受大量的输入信息, 比如输入信息是高清图片时, 输入信息量可能达到上千万, 让神经网络直接从上千万个信息源中学习是一件很吃力的工作. 所以, 何不压缩一下, 提取出原图片中的最具代表性的信息, 缩减输入信息量, 再把缩减过后的信息放进神经网络学习. 这样学习起来就简单轻松了. 所以, 自编码就能在这时发挥作用. 通过将原数据白色的X 压缩, 解压 成黑色的X, 然后通过对比黑白 X ,求出预测误差, 进行反向传递, 逐步提升自编码的准确性. 训练好的自编码中间这一部分就是能总结原数据的精髓. 可以看出, 从头到尾, 我们只用到了输入数据 X, 并没有用到 X 对应的数据标签, 所以也可以说自编码是一种非监督学习. 到了真正使用自编码的时候. 通常只会用到自编码前半部分.

## 参考
- [当我们在谈论 Deep Learning：AutoEncoder 及其相关模型](https://zhuanlan.zhihu.com/p/27865705)
- [什么是自编码 Autoencoder](https://zhuanlan.zhihu.com/p/24813602)
- [深度学习基础：Autoencoders](https://zhuanlan.zhihu.com/p/34201555)
- [Implementing Autoencoders in Keras: Tutorial](https://www.datacamp.com/community/tutorials/autoencoder-keras-tutorial)
- [Keras上实现AutoEncoder自编码器](https://blog.csdn.net/marsjhao/article/details/68928486)
- [Autoencoders](http://ufldl.stanford.edu/tutorial/unsupervised/Autoencoders/)
- [Autoencoder-wiki](https://en.wikipedia.org/wiki/Autoencoder)
