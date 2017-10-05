---
layout:     post
title:      CNN卷积网络的Python实现(四):池化和BN层的实现
date:       2017-10-05 12:01:00
author:     "nickiwei"
header-img: "img/post-bg-2015.jpg"
tags:
    - 深度学习
---


本文是《CNN卷积网络的Python实现》系列的第四篇， 在前文实现的FCN和CNN卷积层的基础上， 实现卷积层后常用的spatial BN层和 池化层。

关于BN的作用， 请参见《CNN卷积网络的Python实现I: FCN网络实现》。

## 池化Pooling的理解
池化其实就是downSampling 下采样， 用以降低数据的规模。 

池化的注意事项：

1， 有时， 池化会保留池化的选择位置。 这是为了在之后的上采样中更准确的还原数据。最常见的应用是Object detection中的上采样层。

2, 池化可以被卷积层选择合适的stride替代， 实际上， 近期的研究表明， 相比池化， 选择stride是更好的方法， 因为他提供了更好的非线性性。

3， 最常见的池化是max 池化（非线性性）， 其余有如平均， 求和等池化（线性）。一般池化层要求选择适当的stride使得采样之间不overlap.

## Naive max Pooling 层的实现

### Forward Pass

```python
def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    Returns a tuple of:
    - out: Output data
    - cache: (x, pool_param)
    """
    N, C, H, W = x.shape
    HH, WW, stride = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']
    Hn = 1 + int((H - HH) / stride)
    Wn = 1 + int((W - WW) / stride)
    out = np.zeros((N, C, Hn, Wn))
    for i in range(Hn):
        for j in range(Wn):
            out[..., i, j] = np.max(x[..., i*stride:i*stride+HH, j*stride:j*stride+WW], axis=(2,3))
    cache = (x, out, pool_param)
    return out, cache
```

很简单， 选择对应的几项， 选出最大值即可。

### Backward Pass

```python
def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    x, out, pool_param = cache
    N, C, Hn, Wn = dout.shape
    HH, WW, stride = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']
    dx = np.zeros_like(x)
    for i in range(Hn):
        for j in range(Wn):
            #(N, C, HH, WW) vs (N, C)
            mark = x[..., i*stride:i*stride+HH, j*stride:j*stride+WW] == out[..., i, j][..., np.newaxis, np.newaxis]
            #(N, C, HH, WW) vs (N, C)
            dx[..., i*stride:i*stride+HH, j*stride:j*stride+WW] = mark * dout[..., i, j][..., np.newaxis, np.newaxis]   
    return dx
```
理解： 
max将梯度传递给最大的那一项， 其余项梯度均为0. 构造一个实现这个逻辑的mark与dout相乘即可。

## Fast max Pooling 层的实现

### Forward Pass

我们同样可以使用im2col的方法来实现pooling， 但他相比naive的实现并没有显著的快。

另外一种reshape的方法可以大幅提升pooling的效率， 但他对输入有较严格的要求。

```python
assert pool_height == pool_width == stride, 'Invalid pool params'
assert H % pool_height == 0
assert W % pool_height == 0
```

我们要求， 池化核必须是正方形且stride保证恰不overlap。同时， H,W恰能被pool_height整除（即池化核恰好不重复的覆盖整个输入）。在实际使用池化算法时， 应尽量保证这些条件得到满足， 避免低效池化。

此时， 我们有如下高速算法：

```python
def max_pool_forward_reshape(x, pool_param):
    """
    A fast implementation of the forward pass for the max pooling layer that uses
    some clever reshaping.

    This can only be used for square pooling regions that tile the input.
    """
    N, C, H, W = x.shape
    pool_height, pool_width = pool_param['pool_height'], pool_param['pool_width']
    stride = pool_param['stride']
    assert pool_height == pool_width == stride, 'Invalid pool params'
    assert H % pool_height == 0
    assert W % pool_height == 0
    x_reshaped = x.reshape(N, C, H // pool_height, pool_height,
                           W // pool_width, pool_width)
    out = x_reshaped.max(axis=3).max(axis=4)

    cache = (x, x_reshaped, out)
    return out, cache
```

虽然难以想象reshape后的形状， 但是可以这么想， 最后分出的四维张量是有若干pw*pw的小方格组成的立方体， 只需要沿着某两个轴取得这些小方格的max即可。

### Backward Pass

针对reshape方法求梯度。

```python
def max_pool_backward_reshape(dout, cache):
    """
    A fast implementation of the backward pass for the max pooling layer that
    uses some clever broadcasting and reshaping.

    This can only be used if the forward pass was computed using
    max_pool_forward_reshape.

    NOTE: If there are multiple argmaxes, this method will assign gradient to
    ALL argmax elements of the input rather than picking one. In this case the
    gradient will actually be incorrect. However this is unlikely to occur in
    practice, so it shouldn't matter much. One possible solution is to split the
    upstream gradient equally among all argmax elements; this should result in a
    valid subgradient. You can make this happen by uncommenting the line below;
    however this results in a significant performance penalty (about 40% slower)
    and is unlikely to matter in practice so we don't do it.
    """
    x, x_reshaped, out = cache

    dx_reshaped = np.zeros_like(x_reshaped)
    out_newaxis = out[:, :, :, np.newaxis, :, np.newaxis]
    mask = (x_reshaped == out_newaxis)
    dout_newaxis = dout[:, :, :, np.newaxis, :, np.newaxis]
    dout_broadcast, _ = np.broadcast_arrays(dout_newaxis, dx_reshaped)
    dx_reshaped = dout_broadcast * mask
    #The line blow is to ensure everyone get correct result
    #dx_reshaped /= np.sum(mask, axis=(3, 5), keepdims=True)
    dx = dx_reshaped.reshape(x.shape)

    return dx
```
理解：

1， 求dx\_reshaped仍然是针对max的mask法。 由于dx_reshape只是对dx形状的变化， 不影响梯度， 所以直接将梯度reshape即可。

2， 注释掉的一行是为了解决有多个值与max值一样大而导致的梯度错误问题， 但在实际实践中， 影响不大却对效率有较大影响。 所以一般不处理。

## spatial BN层的实现

对于CNN网络来说， 每个Channel都被视作一个独立的dimension， 因此， 共有N\*H\*W个dimension, 按照此方向reshape并带入原方法中即可。

注意区别transpose和reshape， 为了保证维度不改变， reshape合并的维度必须相邻且按照指定顺序。反之同理。

### Forward Path
```python
def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """
    Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. 
    """
    N, C, H, W = x.shape
    X_reshaped = x.transpose(0, 2, 3, 1).reshape(-1, C)
    out, cache = batchnorm_forward(x, gamma, beta, bn_param)
    out = out.reshape(N, H, W, C).transpose(0, 1, 2, 3)
    return out, cache
```

### Backward Path

```python
def spatial_batchnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    N, C, H, W = dout.shape
    dout_reshaped = dout.transpose(0, 2, 3, 1).reshape(-1, C)
    dx, dgamma, dbeta = batchnorm_backward(dout_reshaped, cache)
    dx = dx.reshape(N, H, W, C).transpose(0, 1, 2, 3)

    return dx, dgamma, dbeta
```

## 延伸学习
以上四篇， 我们介绍了所有常用的深度网络layer, 下一篇我们将详细介绍如何搭建和fine-tune一个深度CNN网路。
