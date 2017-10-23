---
layout:     post
title: CNN卷积网络的Python实现(一):FCN全连接网络
date:       2017-09-01 12:00:00
author:     "nickiwei"
header-img: "img/post-bg-2015.jpg"
tags:
    - 深度学习
---

这个系列从最基础的全连接网络开始， 从零开始实现包含CNN, RNN等在内的深度网络模型。本文是该系列的第一篇， 介绍全连接网络的实现。

*欢迎转载， 转载请注明出处及链接。*

*完整代码库请查看我的GithubRepo: https://github.com/nick6918/MyDeepLearning. 部分代码参考了Stanford CS231n 课程作业。*

## 为什么要使用神经网络模型

### 1， 用以代替自身不可求或其梯度不易求的函数和分布

我们在很多问题中， 比如强化学习的Q-Learning 算法， 过去， 由于Q方程本身有intractable的部分， 只能使用逼近的办法，但效果不好。 在使用NN去替代传统的Q方程之后， 效果大幅提升。

在使用NN代替原方程之后， 使用NN的梯度代替原方程的梯度， 这大大简化了求梯度的难度。使得梯度自动化成为可能。

### 2， 用来增强模型对非线性的泛化能力

NN是除了SVM的核函数之外， 第二种能够给模型提供非线性泛化能力的模型， 且performance要好于SVM。

### 3， 最重要的， 效果好

经过数学上还难于解释， NN在诸如图像识别， 自然语言处理等方面， 效果非常好。

## FCN Layer的实现

### Forward Pass

```python
def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.
    """
    
    X = x.reshape(x.shape[0], -1) #(N, D)

    out = X.dot(w) + b[np.newaxis, ...]

    cache = (x, w, b)
    return out, cache
```
简单， 做一次矩阵乘法。

### Backward Pass

```python
def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.
    """
    x, w, b = cache
    N = x.shape[0]
    X = x.reshape(N, -1)

    dx = dout.dot(w.T).reshape(x.shape)
    dw = X.T.dot(dout)
    db = dout.T.dot(np.ones(N))

    return dx, dw, db
```
由此可见， 求梯度被大大简化了。

## 激活函数层的实现

以Relu为例。

### Forward Pass

```python
def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).
    """
    out = np.maximum(x, 0)

    cache = x
    return out, cache

```

### Backward Pass

```python
def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).
    """
    dx, x = None, cache
    
    dx = dout
    dx[x<=0] = 0

    return dx
```

## Loss Function的实现

在深度网络的最后， 一般都是一个Loss Function， 用来计算估计损失。 常见的Loss Function有Hinge Loss, Softman Loss等。

### Hinge Loss

![Hinge Loss](http://a2.qpic.cn/psb?/V14QIlwE1OZqS0/d66VSMrgoTvfyLYHwvNoJWMprdvwhRb7soYS.l4ktdg!/b/dOIAAAAAAAAA&bo=BQE1AAAAAAABARc!&rf=viewer_4&t=5)

其中， delta为噪音容限， 通常取1即可。


```python
def svm_loss(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification. Also as Hinge Loss.
    """
    
    N = x.shape[0]
    correct_class_scores = x[np.arange(N), y]
    margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
    margins[np.arange(N), y] = 0
    loss = np.sum(margins) / N
    num_pos = np.sum(margins > 0, axis=1)
    dx = np.zeros_like(x)
    dx[margins > 0] = 1
    dx[np.arange(N), y] -= num_pos
    dx /= N
    return loss, dx
```

同时计算了dLoss_dx, 作为梯度流(Gredient flow)的开始， 以链式法则向后传递。
注意， max（x, 0）的梯度在<0时取0.

### Softmax Loss

![Softmax](http://a2.qpic.cn/psb?/V14QIlwE1OZqS0/q5kHF4J.TUvHilkQ*7RXRtD8EqySngXN9TyAaRFbDRQ!/b/dGwBAAAAAAAA&bo=cwAuAAAAAAABAXs!&rf=viewer_4&t=5)

```python
def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.
    """
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx
```

## 延伸学习
在实现了全连接层， 激活函数层和Loss层之后， 理论上我们就可以搭建一个最简单的神经网络了。 但这个神经网络有诸多问题， 比如， 

1， 当网络越来越深时， 系统难于converge， 容易过拟合或梯度消失。 为此， 我们引入了多个Regulariztaion层。

详见第二篇。

2， 待估参数过多， 非线性泛化能力不足。为此， 我们引入了卷积层。

详见第三篇。

---

## 快速联系作者

欢迎关注我的知乎: https://www.zhihu.com/people/NickWey 


或直接在Github上联系我: https://github.com/nick6918
