
---
layout:     post
title: paddlepaddle深度学习实战(一):浅层模型的搭建
date:       2017-11-28 12:01:00
author:     "nickiwei"
header-img: "img/post-bg-2015.jpg"
tags:
    - 深度学习
---

这个系列从应用paddlepaddle(以下简称paddle)搭建最基础的学习模型开始， 逐个实现各种常用的深度学习应用案例。本系列假设读者已经掌握了[CNN卷积网络的Python实现](http://nickiwei.github.io/2017/09/01/CNN卷积网络的Python实现I-FCN全连接网络/)系列的内容。此外, 本系列中的所有代码均基于python实现, 其他c++相关的实现及源码（包括分布式架构）可见下一系列, coming soon~

欢迎转载， 转载请注明出处及链接。

部分案例及代码参考了百度内部的深度学习课程《PaddlePaddle深度学习实战》，在此一并感谢。

本节我们尝试利用paddle构建几个简单的浅层模型。

## softmax分类器的搭建

```python
def softmax_regression(img):
	#img: (N, D)
	predict = pd.layer.fc(input = img, act = pd.activation.softmax(), size=10)
	return predict
```

很好理解， 一个全连接层接了一个softmax激活函数， 如果读者对此理解有难度， 可以查看我的[CNN卷积网络的Python实现系列](http://nickiwei.github.io/2017/09/01/CNN卷积网络的Python实现I-FCN全连接网络/)。

## FCN全连接分类器

```python
def fcn(img):
	H1 = pd.layer.fc(input = img, size = 128, act = pd.activation.Relu())
	H2 = pd.layer.fc(input = H1, size = 64, act = pd.activation.Relu())
	predict = pd.layer.fc(input = H1, size = 10, act = pd.activation.Softmax())
	return predict
```

还是很好理解， 三层全连接最后softmax分类。在paddle中， 除了基础的fc layer外， 还提供了一种selective layer, 具体api如下：

```python
sel_fc = pd.selective_fc(input=input, size=128, act=paddle.v2.activation.Tanh(), select=mask)
```
其中， mask是一个型如output的布尔矩阵， 用以过滤输出。 

## 浅层CNN的实现

有了FCN的基础， 我们就可以实现一个简单的CNN， 在paddle中的基本conv layer和pooling layer如下：

```python
conv = img_conv(input=data, filter_size=1, filter_size_y=1,
                      num_channels=8,
                      num_filters=16, stride=1,
                      bias_attr=False,
                      act=paddle.v2.activation.Relu()
                      trans=False)    
                      
maxpool = img_pool(input=conv,
                         pool_size=3,
                         pool_size_y=5,
                         num_channels=8,
                         stride=1,
                         stride_y=2,
                         padding=1,
                         padding_y=2,
                         pool_type=MaxPooling())                                 
```
注意几点：

1, 在paddle中， 包含padding, stride, dilation等参数都有几种设置方式， 以padding为例：

```
padding = 1, 表示四周均padding 1个单位。
padding = (1,1, 2, 2), 表示横向padding一个单位， 纵向padding两个单位。
padding = 1, padding_y = 2 等同于 padding = (1, 1, 2, 2)。
```

2, 提供了trans选项， 当trans=True时， 完成反卷机（上采样）。

3, 通过bias\_attr和bias\_shared控制bias的使用和数量， bias\_attr=True， 每个filter分配一个bias， bias_\shared=True, 各个filter共享一个bias.

4, num_channels参数。截取输入矩阵的channel长度， 如果不设置， 则自动读取。暂时搞不清楚这个参数的使用场景。

在paddle中也提供了一种更为简单的conv\_pool\_layer在networks lib中， 使用如下：

```python
conv_pool_1 = pd.networks.simple_img_conv_pool(
		input = img, 
		filter_size = 5,
		num_filters = 20,
		num_channel = 1,
		pool_size = 2,
		pool_stride = 2,
		act = pd.activation.Relu() 
		)	
```

可见， 基本上就是合并了conv 和 pooling两个layer的接口。基于以上接口， 我们就可以构造一个简单的cnn分类器了.

```python
def cnn(img):
	#img: (N, C, H, W)
	conv_pool_1 = pd.networks.simple_img_conv_pool(
		input = img, 
		filter_size = 5,
		num_filters = 20,
		num_channel = 1,
		pool_size = 2,
		pool_stride = 2,
		act = pd.activation.Relu() 
		)	
	conv_pool_2 = pd.networks.simple_img_conv_pool(
		input = conv_pool_1, 
		filter_size = 5,
		num_filters = 50,
		num_channel = 20,
		pool_size = 2,
		pool_stride = 2,
		act = pd.activation.Relu()
		)
	predict = pd.layer.fc(input = conv_pool_2, size = 10, act = pd.activation.Softmax())
	return predict
```

除了基本的conv layer外， paddle中还提供了一些特殊的conv api。如conv\_
op, conv\_project, conv\_shift等， 具体可查看paddle这一部分的文档: http://www.paddlepaddle.org/docs/develop/documentation/zh/api/v2/config/layer.html

# 延伸学习

下一节我们将利用本节所搭建的浅层网络 实现MNIST数据集上的数字识别。 第三节中， 我们将开始介绍一些深层网络。

# 快速联系作者

欢迎关注我的知乎: https://www.zhihu.com/people/NickWey

或直接在Github上联系我: https://github.com/nick6918
