---
layout:     post
title: paddlepaddle深度学习实战(一):浅层模型的搭建
date:       2017-11-28 12:04:00
author:     "nickiwei"
header-img: "img/post-bg-2015.jpg"
tags:
    - 深度学习
---

这个系列从应用paddlepaddle(以下简称paddle)搭建最基础的学习模型开始， 逐个实现各种常用的深度学习应用案例。本系列假设读者已经掌握了[CNN卷积网络的Python实现](http://nickiwei.github.io/2017/09/01/CNN卷积网络的Python实现I-FCN全连接网络/)系列的内容。此外, 本系列中的所有代码均基于python实现, 其他c++相关的实现及源码（包括分布式架构）可见下一系列, coming soon~

欢迎转载， 转载请注明出处及链接。

部分案例及代码参考了百度内部的深度学习课程《PaddlePaddle深度学习实战》，在此一并感谢。

## 问题定义

我们希望实现一个这样的电影推荐系统。

```
输入：
movie_id: 1234
user_id: 1

输出:
prediction_score: 4.25(0-5)
```

## 文本卷积神经网络

#TODO


## 延伸阅读

下一节我们将实现一个基于paddle的简单的推荐系统。 从下下一节开始， 我们将进入NLP部分。

# 快速联系作者

欢迎关注我的知乎: https://www.zhihu.com/people/NickWey

或直接在Github上联系我: https://github.com/nick6918


