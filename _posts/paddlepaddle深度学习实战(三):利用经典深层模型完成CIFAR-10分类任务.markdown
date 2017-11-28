这个系列从应用paddlepaddle(以下简称paddle)搭建最基础的学习模型开始， 逐个实现各种常用的深度学习应用案例。本系列假设读者已经掌握了[CNN卷积网络的Python实现](http://nickiwei.github.io/2017/09/01/CNN卷积网络的Python实现I-FCN全连接网络/)系列的内容。此外, 本系列中的所有代码均基于python实现, 其他c++相关的实现及源码（包括分布式架构）可见下一系列, coming soon~

欢迎转载， 转载请注明出处及链接。

部分案例及代码参考了百度内部的深度学习课程《PaddlePaddle深度学习实战》，在此一并感谢。

本节我们尝试使用一些经典的深层模型处理CIFAR-10上的分类问题。 注意，我们将使用到vgg和resnet两个模型， 如果对此不太了解， 请自行查阅论文。

## vgg模型的构造

这里我们给出一个paddle中vgg net的具体实现。

```python
def vgg_bn_drop(input):
    def conv_block(ipt, num_filter, groups, dropouts, num_channels=None):
        return paddle.networks.img_conv_group(
            input=ipt,
            num_channels=num_channels,
            pool_size=2,
            pool_stride=2,
            conv_num_filter=[num_filter] * groups,
            conv_filter_size=3,
            conv_act=paddle.activation.Relu(),
            conv_with_batchnorm=True,
            conv_batchnorm_drop_rate=dropouts,
            pool_type=paddle.pooling.Max())

    conv1 = conv_block(input, 64, 2, [0.3, 0], 3)
    conv2 = conv_block(conv1, 128, 2, [0.4, 0])
    conv3 = conv_block(conv2, 256, 3, [0.4, 0.4, 0])
    conv4 = conv_block(conv3, 512, 3, [0.4, 0.4, 0])
    conv5 = conv_block(conv4, 512, 3, [0.4, 0.4, 0])

    drop = paddle.layer.dropout(input=conv5, dropout_rate=0.5)
    fc1 = paddle.layer.fc(input=drop, size=512, act=paddle.activation.Linear())
    bn = paddle.layer.batch_norm(
        input=fc1,
        act=paddle.activation.Relu(),
        layer_attr=paddle.attr.Extra(drop_rate=0.5))
    fc2 = paddle.layer.fc(input=bn, size=512, act=paddle.activation.Linear())
    return fc2
```

这里使用了一个特殊的networks lib layer, 

```python
paddle.networks.img_conv_group(
            input=ipt,
            num_channels=num_channels,
            pool_size=2,
            pool_stride=2,
            conv_num_filter=[num_filter] * groups,
            conv_filter_size=3,
            conv_act=paddle.activation.Relu(),
            conv_with_batchnorm=True,
            conv_batchnorm_drop_rate=dropouts,
            pool_type=paddle.pooling.Max())
```

理解也很简单， 就是多个conv层后带一个pooling layer.

## ResNet模型的构造

这里我们介绍几个ResNet在实现上的细节：

```python
def shortcut(ipt, n_in, n_out, stride):
    if n_in != n_out:
        return conv_bn_layer(ipt, n_out, 1, stride, 0,
                             paddle.activation.Linear())
    else:
        return ipt
        
def basicblock(ipt, ch_out, stride):
    ch_in = ch_out * 2
    tmp = conv_bn_layer(ipt, ch_out, 3, stride, 1)
    tmp = conv_bn_layer(tmp, ch_out, 3, 1, 1, paddle.activation.Linear())
    short = shortcut(ipt, ch_in, ch_out, stride)
    return paddle.layer.addto(input=[tmp, short], act=paddle.activation.Relu())

def layer_warp(block_func, ipt, features, count, stride):
    tmp = block_func(ipt, features, stride)
    for i in range(1, count):
        tmp = block_func(tmp, features, 1)
    return tmp
```

有了以上几个支持方法， 我们就可以方便的实现一个resnet.

```python
def resnet_cifar10(ipt, depth=32):
    # depth should be one of 20, 32, 44, 56, 110, 1202
    assert (depth - 2) % 6 == 0
    n = (depth - 2) / 6
    nStages = {16, 64, 128}
    conv1 = conv_bn_layer(
        ipt, ch_in=3, ch_out=16, filter_size=3, stride=1, padding=1)
    res1 = layer_warp(basicblock, conv1, 16, n, 1)
    res2 = layer_warp(basicblock, res1, 32, n, 2)
    res3 = layer_warp(basicblock, res2, 64, n, 2)
    pool = paddle.layer.img_pool(
        input=res3, pool_size=8, stride=1, pool_type=paddle.pooling.Avg())
    return pool
```

## 模型训练与测试

模型的训练与测试与上一章完全一致， 唯一的不同在于handler的微调。 读者可自行查阅项目代码了解细节。

## 延伸阅读

下一节我们将实现一个基于paddle的简单的推荐系统。 从下下一节开始， 我们将进入NLP部分。

# 快速联系作者

欢迎关注我的知乎: https://www.zhihu.com/people/NickWey

或直接在Github上联系我: https://github.com/nick6918
