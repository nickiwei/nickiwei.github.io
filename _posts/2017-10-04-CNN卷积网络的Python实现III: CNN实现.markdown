---
layout:     post
title:CNN卷积网络的Python实现(三):卷积网络实现
date:       2017-10-04 12:00:00
author:     "nickiwei"
header-img: "img/post-bg-2015.jpg"
tags:
    - 深度学习
---


在简要说明为什么要引入卷积网络之后， 利用Numpy 实现了naive版本的卷积网络的forward pass核backward pass， 并介绍了快速卷积网络的实现。

*本文中的部分代码来自于Stanford CS231n 课程作业。*


## 卷积网络的优势

卷积神经网络CNN是深度学习中最常用的拓扑结构， 相比于传统的FCN(Fully Connected Network), CNN的好处是显而易见的， 在我的理解中， CNN的优势主要在于：

<b>1， 卷积核共享参数Shared parameters</b>

可以将卷积网络的卷积核看作是一种共享参数， 在FCN中, 每一个Neuron都对应一个待估参数， 产生庞大的参数群， 使得网络难于converge. 举例来说, AlexNet共有62M待估参数， 其中56M参数都来自于最后一层的全连接层。 如何尽可能避免全连接层， 也成为CNN网络拓扑结构设计中的一个重要目标。

<b>2， 更好的非线性泛化(generalization)能力</b>
传统的FCN依靠激活函数提供非线性泛化能力， 在CNN中， 由于卷积操作本身也是一种非线性函数， 因此， CNN也获得了更好的非线性泛化能力， 效率上更易converge， performance也更优。

<b>3， 倾向于更小的卷积核和更深的网络结构</b>

深层FCN不一定带来更好的泛化能力， 但实证证明， 深层CNN确实能够产生更好的泛化能力。

一方面， 卷积核共享参数和Batch Normalizaton技术使得深层网络的实现成为可能， 另一方面， CNN的深层网络也确实提供了更好的泛化能力。可以从以下角度理解这一点， 对于3层3\*3卷积核的网络和1层7\*7的网络， 二者的效果(receptive region)是一致的， 但实验证明， 3层3\*3网络提供了更好的泛化能力， 同时也进一步减少了所需参数（3\*3*3=27 < 7\*7=49）。

## Naive 版本卷积层Layer的实现

### Forward Pass

Forward Pass中最重要的就是理解， 结果tensor与原tensor的对应关系。如下图：

[图片]

这个我就不多解释了， 如果理解有困难， 可自行查看一些资料。下面上代码：

```python
def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width HH.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    N, C, H, W = x.shape
    F, C, HH,WW= w.shape
    pad = conv_param["pad"]
    stride = conv_param["stride"]
    X = np.pad(x, ((0,0), (0, 0), (pad, pad),(pad, pad)), 'constant')
    
    Hn = 1 + int((H + 2 * pad - HH) / stride)
    Wn = 1 + int((W + 2 * pad - WW) / stride)
    out = np.zeros((N, F, Hn, Wn))
    for n in range(N):
        for m in range(F):
            for i in range(Hn):
                for j in range(Wn):
                    data = X[n, :, i*stride:i*stride+HH, j*stride:j*stride+WW].reshape(1, -1)
                    filt = w[m].reshape(-1, 1)
                    out[n, m, i, j] = data.dot(filt) + b[m]
    cache = (x, w, b, conv_param)
    return out, cache
```

代码比较好理解， 卷积结果中的每一项， 对应于核与原tensor中如下子tensor做卷积。

	X[n, :, i\*stride:i\*stride+HH, j\*stride:j*stride+WW]

### Backward Pass

要理解到， 神经网络的一个重要特性就是， 用卷积网络逼近复杂的函数， 从而用相对简单的神经网络梯度， 代替复杂的分析函数的梯度。 这里实现了最简单的卷积网络的梯度：

```python
def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    x, w, b, conv_param = cache
    N, F, Hn, Wn = dout.shape
    N, C, H, W = x.shape
    F, C, HH,WW= w.shape
    pad = conv_param["pad"]
    stride = conv_param["stride"]
    dw = np.zeros_like(w)
    X = np.pad(x, ((0,0), (0, 0), (pad, pad),(pad, pad)), 'constant')
    dX = np.zeros_like(X)
    for n in range(N):
        for m in range(F):
            for i in range(Hn):
                for j in range(Wn):
                    dX[n, :, i*stride:i*stride+HH, j*stride:j*stride+WW] += w[m] *  dout[n, m, i, j]
                    dw[m] += X[n, :, i*stride:i*stride+HH, j*stride:j*stride+WW] *  dout[n, m, i, j]
    db = np.sum(dout, axis=(0, 2, 3))
    dx = dX[:, :, pad:-pad, pad:-pad]
    return dx, dw, db

```

假设卷积核大小为Hw*Ww

在合适的padding下， 原tensor中的每一项都需要与卷积核的每一项做卷积， 贡献给结果tensor中的某一项， 反之， 原tensor的梯度是Hw\*Ww项结果tensor的核。

读者可以自行理解一下。

## Fast 版本卷积层Layer的实现

在理解了naive版本的卷积实现后， 我们开始考虑加速卷积。这一部分是本文的难点所在。

### Forward Pass

优化的核心还是老套路，考虑用矩阵乘法代替多重for循环， 从而降低运算复杂度。希望构造一个更大的矩阵， 使得该矩阵的每一列和向量化的卷积核的内积。

Step 1, 我们将F个卷积核合并成一个更大的卷积核tensor, 新的大卷积核WW的shape = (Hw\*Ww\*C， F). 

Step 2, 将(N, C, W+2F, H+2F) 按照stride的步长取C\*Hw\*Ww项作为新矩阵的一行， 可以看到， 共有N\*Hw\*Nw项（可以这样理解， 我们取出的每一个行向量与都对应着一个输出矩阵中的一项， 所以共有N\*Hw\*Nw项) ， 此时新矩阵XX的shape = （N\*Hn\*Wn, Hw\*Ww*C）

```python
#此步骤的API
x_cols = im2col_cython(x, Ww, Hw, pad, stride)
```

Step 3, 计算XX * WW, 结果Out的shape为(N\*Hn*Wn, F), 重排即可得输出矩阵(N, F, Hn, Wn)

基于此思路实现加速版本的卷积

```python
def conv_forward_im2col(x, w, b, conv_param):
    """
    A fast implementation of the forward pass for a convolutional layer
    based on im2col and col2im.
    """
    N, C, H, W = x.shape
    num_filters, _, filter_height, filter_width = w.shape
    stride, pad = conv_param['stride'], conv_param['pad']

    # Check dimensions
    assert (W + 2 * pad - filter_width) % stride == 0, 'width does not work'
    assert (H + 2 * pad - filter_height) % stride == 0, 'height does not work'

    # Create output
    out_height = (H + 2 * pad - filter_height) // stride + 1
    out_width = (W + 2 * pad - filter_width) // stride + 1
    out = np.zeros((N, num_filters, out_height, out_width), dtype=x.dtype)

    # x_cols = im2col_indices(x, w.shape[2], w.shape[3], pad, stride)
    x_cols = im2col_cython(x, w.shape[2], w.shape[3], pad, stride)
    res = w.reshape((w.shape[0], -1)).dot(x_cols) + b.reshape(-1, 1)

    out = res.reshape(w.shape[0], out.shape[2], out.shape[3], x.shape[0])
    out = out.transpose(3, 0, 1, 2)

    cache = (x, w, b, conv_param, x_cols)
    return out, cache
```

注意事项

	im2col_cython(x, Ww, Hw, pad, stride)

该API的实现，构造了一个新的矩阵， 该矩阵大小为N\*Hn\*Wn\*Hw\*Ww*C， 相比于原矩阵的大小(N\*C\*W\*H), 放大倍数是相当可观的。是牺牲空间换时间的算法。

为加速实现， 有以byte为单位的trick如下：

```python
def im2col_cython(x, WW, HH, pad, stride):
    shape = (C, HH, WW, N, out_h, out_w)
    strides = (H * W, W, 1, C * H * W, stride * W, stride)
    strides = x.itemsize * np.array(strides)
    x_stride = np.lib.stride_tricks.as_strided(x_padded,
                  shape=shape, strides=strides)
    x_cols = np.ascontiguousarray(x_stride)
    x_cols.shape = (C * HH * WW, N * out_h * out_w)
    return x_cols
```

### Backward Pass

```python
def conv_backward_im2col(dout, cache):
    """
    A fast implementation of the backward pass for a convolutional layer
    based on im2col and col2im.
    """
    x, w, b, conv_param, x_cols = cache
    stride, pad = conv_param['stride'], conv_param['pad']

    db = np.sum(dout, axis=(0, 2, 3))

    num_filters, _, filter_height, filter_width = w.shape
    dout_reshaped = dout.transpose(1, 2, 3, 0).reshape(num_filters, -1)
    dw = dout_reshaped.dot(x_cols.T).reshape(w.shape)

    dx_cols = w.reshape(num_filters, -1).T.dot(dout_reshaped)
    # dx = col2im_indices(dx_cols, x.shape, filter_height, filter_width, pad, stride)
    dx = col2im_cython(dx_cols, x.shape[0], x.shape[1], x.shape[2], x.shape[3],
                       filter_height, filter_width, pad, stride)

    return dx, dw, db
```

注意事项：
如何理解dx_cols 到dx, 即函数 

	dx = col2im_cython(dx_cols, x.shape[0], x.shape[1], x.shape[2], x.shape[3], filter_height, filter_width, pad, stride)
	
在dx_cols中，每个x出现了Hw*Ww次， dx需要将他们加在一起。

## 延伸学习

除了卷积层的实现外， CNN还包括了spatial BN层和池化层两个常用layer。可参考：本系列第四篇。
