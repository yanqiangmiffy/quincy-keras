## keras 损失函数

- mean_squared_error
- mean_absolute_error
- mean_absolute_percentage_error
- mean_squared_logarithmic_error
- squared_hinge
- hinge
- categorical_hinge
- logcosh
- categorical_crossentropy
- sparse_categorical_crossentropy
- binary_crossentropy
- kullback_leibler_divergence
- poisson
- cosine_proximity

## 预定义激活函数
- softmax：对输入数据的最后一维进行softmax，输入数据应形如(nb_samples, nb_timesteps, nb_dims)或(nb_samples,nb_dims)
- elu:指数线性单元。
- selu: 可伸缩的指数线性单元（Scaled Exponential Linear Unit），参考Self-Normalizing Neural Networks
- softplus：Softplus 激活：log(exp(x) + 1)。
- softsign：Softsign 激活：x / (abs(x) + 1)。
- relu：线性修正单元。
- tanh：双曲正切激活函数。
- sigmoid
- hard_sigmoid：Hard sigmoid 激活函数，计算速度比 sigmoid 激活函数更快。
- linear：线性激活函数（即不做任何改变）