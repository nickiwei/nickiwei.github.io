---
layout:     post
title: CNN卷积网络的Python实现(二):Regularization正则化实现
date:       2017-09-04 12:01:00
author:     "nickiwei"
header-img: "img/post-bg-2015.jpg"
tags:
    - 深度学习
---

这个系列从最基础的全连接网络开始， 从零开始实现包含CNN, RNN等在内的深度网络模型。本文是该系列的第二篇， 介绍正则化的原理及实现。

欢迎转载， 转载请注明出处及链接。

完整代码库请查看我的GithubRepo: https://github.com/nick6918/MyDeepLearning. 部分代码参考了Stanford CS231n 课程作业。

## Regularization 的作用
深度学习模型同样存在过拟合问题， 为了防止过度拟合， 我们在估计参数, 设计模型, 样本处理时， 均可以设法引入regularization, 抑制过拟合。

在参数估计时， 常使用L2 Regualization, 即

![L2Reg](https://github.com/nickiwei/nickiwei.github.io/blob/master/img/l2r.gif)

除此之外， 还有L1和smooth L1-L2拟合(beta*L2+L1)

同时， 在模型设计时， 常添加DropOut层以增加Regularization, 同时， BatchNormalization也可被视作一种Regularization.

最后， 我们还可以在样本数据预处理时， 通过引入Data Augmentation, 增强Regularization, 提高test performance, 防止过拟合。

## DropOut层的理解

我们知道， 当待估参数过多， 模型较复杂且训练样本较少时， 容易产生过拟合现象。如图， 之所以dropout可以抑制overfitting, 是因为在训练阶段， 我们引入了随机性(随机cancel一些Neuron), 在测试阶段， 我们去除掉随机性， 并通过期望的方式应用测试数据。

![dropout](https://github.com/nickiwei/nickiwei.github.io/blob/master/img/DO.png)

为了简化计算， 我们考虑一个只有两个Neuron的神经网络， 且任意一个Neuron在训练阶段被dropout的概率为1/2， 在测试阶段不dropout.

![dropout_Calculation](https://github.com/nickiwei/nickiwei.github.io/blob/master/img/et.png)

可见， 测试数据的最终期望与训练数据的最终期望， 相差一个概率p.为了保证测试数据的一致性， 我们在训练数据的最后除以一个p， 这被称为反向DropOut.

## Inverse DropOut层的实现

### Forward Path

```python
def dropout_forward(x, dropout_param):
    """
    Performs the forward pass for (inverted) dropout.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We drop each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.
    """
    p, mode = dropout_param['p'], dropout_param['mode']
    if 'seed' in dropout_param:
        np.random.seed(dropout_param['seed'])

    N, D = x.shape
    mask = np.random.rand(N, D) <= p
    out = None

    if mode == 'train':
        out = x * mask / p   
    elif mode == 'test':       
        out = x

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache
```

### Backward Path

```python
def dropout_backward(dout, cache):
    """
    Perform the backward pass for (inverted) dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param['mode']

    if mode == 'train':
        dx = dout * mask / dropout_param['p'] 
    elif mode == 'test':
        dx = dout
    return dx
```
反向传播将梯度传递给那些未被cancel的Neuron, 其余梯度均为0.

## BatchNormalization 层的实现

对于常用的激活函数而言， tanh和sigmoid在正负无穷大处均发生饱和， 梯度消失。 而relu则在副半轴梯度消失， 正半轴则会发生梯度爆炸（想象每一层relu，dw都会乘上当层输入x， 随着深度变深， dw变得非常大）。这使得较深层的网络越来越难于收敛。

为了解决这个问题，我们引入了BatchNormalization.

![BN](https://github.com/nickiwei/nickiwei.github.io/blob/master/img/bn.png)

### 如何理解BN

1, 前三个公式， 可以看出， 我们用xhat 代替x， 实在对每一层的输入做白化， 以确保输入不偏移到饱和区。

2， 但彻底的白化意味着由激活函数所引入的非线性性也被抵消了。 因此，我们引入最后一个公式， 用学习参数gamma和beta来scale白化数据xhat.

综上， BN的实质是在确保输入数据不进入饱和区和保证数据的非线性性之间做平衡， 而平衡点在哪里则有系统自己学习。

3, 所谓Batch Normalization而不是Normalization。在训练阶段， 每次输入一个batch(与Batch SGD)保持一致， 进行如上的白化， 并维护一个全局的running mean 和 running std, 这两项随着每次batch的更新而更新。 在测试阶段， 我们直接用running mean 和 running std来做白化。

### Forward Pass 

Forward pass就是逐个实现上述四个公式， 但为了便于求梯度， 我们用CG(Computation Graph)的形式细化上述四个公式。


![BN_CG](https://github.com/nickiwei/nickiwei.github.io/blob/master/img/xhat.png)

如上图， 我们计算出了xhat， 最终的输出out = gamma*xhat + beta, 代码如下:

```python
def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

    if mode == 'train':
        xmean = np.mean(x, axis=0)
        xcentered = x - xmean
        xvar = np.mean(xcentered**2, axis=0)
        xstd = np.sqrt(xvar + eps)
        xhat = xcentered / xstd
        out = gamma * xhat + beta
        cache = xcentered, xstd, gamma

        running_mean = momentum * running_mean + (1 - momentum) * xmean
        running_var = momentum * running_var + (1 - momentum) * xvar
      

    elif mode == 'test':
        
        x = (x - running_mean) / (np.sqrt(running_var) + eps)
        out = gamma * x + beta
        cache = None
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

    return out, cache

```

### Backward Pass

有了CG形式的forward pass之后， 我们就可以比较容易的逐步计算所需的梯度了。

```python

def batchnorm_backward(dout, cache):
    """
    Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    xcentered, xstd, gamma = cache
    N, D = xcentered.shape

    # debug: see middle dx 
    # dLoss_dstd = dout
    # dL1_dxcentered = 0

    dLoss_dbeta = np.ones(N).dot(dout)
    dLoss_dgamma = np.ones(N).dot(dout * xcentered / xstd)
    
    dLoss_dxhat = dout*gamma
    dLoss_stdinv = np.ones(N).dot(xcentered * dLoss_dxhat)
    dL1_dxcentered = dLoss_dxhat/xstd
    dLoss_dstd = -dLoss_stdinv / xstd**2
    dLoss_dvar = 0.5 * dLoss_dstd/ xstd
    dLoss_dxcenteredsq = dLoss_dvar*np.ones((N, D)) / N
    dL2_dxcentered = 2 * dLoss_dxcenteredsq * xcentered
    dLoss_dxcentered = dL1_dxcentered + dL2_dxcentered
    dLoss_dmean = np.ones(N).dot(dLoss_dxcentered)
    dL1_dx = -(dLoss_dmean * np.ones((N, D))) / N
    dL2_dx = dLoss_dxcentered
    dLoss_dx = dL1_dx + dL2_dx

    return dLoss_dx, dLoss_dgamma, dLoss_dbeta
```

## 为什么BatchNormalization也是一种Regularization

在dropout部分， 我们已经解释过， 之所以dropout可以抑制overfitting, 是因为在训练阶段， 我们引入了随机性(随机cancel一些Neuron), 在测试阶段， 我们去除掉随机性， 并通过期望的方式应用测试数据。

在BatchNormalization中， 测试阶段， 我们随机选取了Batch计算running_mean等， 在测试阶段， 应用这些训练参数去除随机性。 因此， BatchNormalization也提供了Regularization的作用， 实际应用中证明， NB在防止过拟合方面确实也有相当好的表现。

## 更多Regularization

### Data Augmentation数据增强

数据增强是一种原理很直观的sample regularization， 在训练阶段， 随机的选择若干个图片切片训练 or 随机的选择 图片反转， 提亮， 拉伸等多种操作的若干种， 测试阶段正常得到结果。

这种方式， 我们可以屏蔽掉一些直观上很容易理解的过拟合现象， 比如不认识照镜子的猫等。

在ResNet的训练中， 使用了这种方案：

1， resize image at {224, 256, 384, 480, 640}

2, For each size, use 10 224 x 224 crops: 4 corners + center, + flips

### Drop Layer

DropOut的一种改进，对于深层网络，可以在训练时随机的drop掉某些layer。 

### 共性

所有的Regularization结构在拓扑结构上都存在某种共性， 即：

Step 1, 在训练阶段， 通过引入随机性， 使得训练不能完美匹配。

Step 2, 在测试阶段， 需要Marginalize排除掉这种随机性(求期望等)

## 延伸学习

在本系列 第四篇会介绍 卷积网络对BN的改进算法: Spatial BN

完整的一个深度网络将会在 本系列第五篇实现。

---

## 快速联系作者

欢迎关注我的知乎: https://www.zhihu.com/people/NickWey 


或直接在Github上联系我: https://github.com/nick6918
