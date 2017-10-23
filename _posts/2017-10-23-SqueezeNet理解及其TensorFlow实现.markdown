---
layout:     post
title: SqueezeNet理解及其TensorFlow实现
date:       2017-09-25 12:01:00
author:     "nickiwei"
header-img: "img/post-bg-2015.jpg"
tags:
    - 深度学习
    - SqeezeNet
    - TensorFlow
---

*欢迎转载， 转载请注明出处及链接。*

*完整代码库请查看我的GithubRepo: <https://github.com/nick6918/MyDeepLearning> .部分代码参考了Stanford CS231n 课程作业。*

## 背景

AlexNet在12年大幅提高了ImageNet识别的准确率， 但AlexNet最大的问题是参数多， 难于收敛和调优。 而SqueezeNet针对此问题， 大幅度减少了待估参数。

同时， SqueezeNet的设计思路是一个很好的缩小模型大小的方法， 可泛化推广。

附SqueezeNet论文： ![SqueezeNet论文]https://arxiv.org/pdf/1602.07360.pdf， 本文的图片均引用自该论文。

## Fire Module

Fire Module是SqueezeNet的核心。

![SqueezeNet](/Img/sn.png)

如图， 一个Fire Module是两层卷积层， 第一层卷积使用1\*1的核S1x1个。 第二层使用1\*1核和3\*3核各e1x1 和 e3x3个。

假设输入矩阵为(N, C, H, W)则Squeeze后的中间矩阵为(N, S1, H, W), 最终的输出矩阵为(N, e1+e3, H, W).

不妨令 

	C = 3, S1 = 3, e1 = e3 = 4

则最终输出为(N, 8, H, W), 所需的核参数为 

	N*C*S1 + N*S1*(e1+9e3) = 129N

考虑达到同样效果得到(N, 8, H, W)的一层3\*3卷积， 核参数个数为

	N*C*9*8 = 216N
	
核参数减少约50%. 假设C=10, 则两个数据变为: 150N/720N, 核参数减少80%. 可见， 相比一个普通的3*3卷积网络， Fire Module在得到相同结果的同时， 能减少50%以上的核参数数量， 且通道越多， 效果越明显。

TF实现如下：

```python
def fire_module(x,inp,sp,e11p,e33p):
    with tf.variable_scope("fire"):
        #(N, inp, H, W) -> (N, sp, H, W)
        with tf.variable_scope("squeeze"):
            W = tf.get_variable("weights",shape=[1,1,inp,sp])
            b = tf.get_variable("bias",shape=[sp])
            s = tf.nn.conv2d(x,W,[1,1,1,1],"VALID")+b
            s = tf.nn.relu(s)
        #(N, e11p, H, W) + (N, e33p, H, W)
        with tf.variable_scope("e11"):
            W = tf.get_variable("weights",shape=[1,1,sp,e11p])
            b = tf.get_variable("bias",shape=[e11p])
            e11 = tf.nn.conv2d(s,W,[1,1,1,1],"VALID")+b
            e11 = tf.nn.relu(e11)
        with tf.variable_scope("e33"):
            W = tf.get_variable("weights",shape=[3,3,sp,e33p])
            b = tf.get_variable("bias",shape=[e33p])
            e33 = tf.nn.conv2d(s,W,[1,1,1,1],"SAME")+b
            e33 = tf.nn.relu(e33)
        return tf.concat([e11,e33],3)
```

## 完整Architecture

![SqueezeNetArchitecture](/Img/sqarc.png)

如图， 基本的SqueezeNet在一个卷积层后跟了8个fireModule, 最后又是一个卷积。
改进的SqueezeNet还增加了Fast Pass.

根据原论文， 有如下注意事项：

1, 卷积核的数量逐步递增。

2，在conv1, fire4, fire8和conv10层有stride的2的max pooling层。

3， 在fire9后有50%的dropout层。

4， 采用了NiN的无FCN设计。

Simple版本实现如下

```python
class SqueezeNet(object):
    def extract_features(self, input=None, reuse=True):
        if input is None:
            input = self.image
        x = input
        layers = []
        with tf.variable_scope('features', reuse=reuse):
            with tf.variable_scope('layer0'):
                W = tf.get_variable("weights",shape=[3,3,3,64])
                b = tf.get_variable("bias",shape=[64])
                x = tf.nn.conv2d(x,W,[1,2,2,1],"VALID")
                x = tf.nn.bias_add(x,b)
                layers.append(x)
            with tf.variable_scope('layer1'):
                x = tf.nn.relu(x)
                layers.append(x)
            with tf.variable_scope('layer2'):
                x = tf.nn.max_pool(x,[1,3,3,1],strides=[1,2,2,1],padding='VALID')
                layers.append(x)
            with tf.variable_scope('layer3'):
                x = fire_module(x,64,16,64,64)
                layers.append(x)
            with tf.variable_scope('layer4'):
                x = fire_module(x,128,16,64,64)
                layers.append(x)
            with tf.variable_scope('layer5'):
                x = tf.nn.max_pool(x,[1,3,3,1],strides=[1,2,2,1],padding='VALID')
                layers.append(x)
            with tf.variable_scope('layer6'):
                x = fire_module(x,128,32,128,128)
                layers.append(x)
            with tf.variable_scope('layer7'):
                x = fire_module(x,256,32,128,128)
                layers.append(x)
            with tf.variable_scope('layer8'):
                x = tf.nn.max_pool(x,[1,3,3,1],strides=[1,2,2,1],padding='VALID')
                layers.append(x)
            with tf.variable_scope('layer9'):
                x = fire_module(x,256,48,192,192)
                layers.append(x)
            with tf.variable_scope('layer10'):
                x = fire_module(x,384,48,192,192)
                layers.append(x)
            with tf.variable_scope('layer11'):
                x = fire_module(x,384,64,256,256)
                layers.append(x)
            with tf.variable_scope('layer12'):
                x = fire_module(x,512,64,256,256)
                layers.append(x)
        return layers

    def __init__(self, save_path=None, sess=None):
        """Create a SqueezeNet model.
        Inputs:
        - save_path: path to TensorFlow checkpoint
        - sess: TensorFlow session
        - input: optional input to the model. If None, will use placeholder for input.
        """
        self.image = tf.placeholder('float',shape=[None,None,None,3],name='input_image')
        self.labels = tf.placeholder('int32', shape=[None], name='labels')
        self.layers = []
        x = self.image
        self.layers = self.extract_features(x, reuse=False)
        self.features = self.layers[-1]
        with tf.variable_scope('classifier'):
            with tf.variable_scope('layer0'):
                x = self.features
                self.layers.append(x)
            with tf.variable_scope('layer1'):
                W = tf.get_variable("weights",shape=[1,1,512,1000])
                b = tf.get_variable("bias",shape=[1000])
                x = tf.nn.conv2d(x,W,[1,1,1,1],"VALID")
                x = tf.nn.bias_add(x,b)
                self.layers.append(x)
            with tf.variable_scope('layer2'):
                x = tf.nn.relu(x)
                self.layers.append(x)
            with tf.variable_scope('layer3'):
                x = tf.nn.avg_pool(x,[1,13,13,1],strides=[1,13,13,1],padding='VALID')
                self.layers.append(x)
        self.classifier = tf.reshape(x,[-1, NUM_CLASSES])

        if save_path is not None:
            saver = tf.train.Saver()
            saver.restore(sess, save_path)
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(self.labels, NUM_CLASSES), logits=self.classifier))

``` 

其他都很好理解， 关注一下fire module核参数的逐步提高即可。

## SqueezeNet Performance

![SqueezeNetP1](/Img/sq.png)

![SqueezeNetP1](/Img/sqmore.png)

详细的performance细节及测试请查看论文。 这里简单的列一些核心结论， 由FIGURE 3可见， Squeeze使用了1/3的核参数的前提下， 基本保持了80%以上的accuracy， 且fast pass能略微提高准确率， complex fast pass则效果不明显。

## 结论

SqueezeNet发表于2017年， 在几乎不降低accuracy的前提下， 大幅减少了参数数量， 主要方法有两者：

1， 借助NiN的无FCN设计取消了全部全连接层。

2， 在CNN层中， 用特别的Fire Module代替一般3*3卷积层。

其主要目的是帮助模型快速收敛， 但与ResNet/DenseNet的思路（加速gredient流动）不同，可以相互补充。 事实上， SqueezeNet也确实在改进版中加入了 fast pass, 但效果不显著。

---

## 快速联系作者

欢迎关注我的知乎: <https://www.zhihu.com/people/NickWey> 


或直接在Github上联系我: <https://github.com/nick6918> 
