---
layout:     post
title: 千亿特征的离散DNN模型探秘（二）：模型训练
date:       2020-02-20 12:01:00
author:     "nickiwei"
header-img: "img/post-bg-2015.jpg"
tags:
    - 机器学习
    - 深度学习
    - 广告算法
---

上一篇中， 我们介绍了离散特征的构造， 同时， 根据是否包含id类特征， 我们介绍了termDNN和IdDNN两种DNN模型。 在这一篇中， 我们将对比这两种模型的区别， 并介绍模型训练和调优。

## TermDNN和IdDNN的区别

如前所属， IdDNN和TermDNN最本质的区别是是否包含含有Cross Id的特征。 但由此延伸，二者产生了更多的区别。

首先， TermDNN是一个可以持久化保存服务的模型， 如前所述， 每一个term交叉特征最终都会训练得到一个embedding， 这个embedding表示什么含义呢？ 他就表示这个term交叉特征的语意， 这个语意特征是相对固定的， 因为term的含义不会突然改变， iphone不会一会儿表示手机， 一会儿表示别的。同时， term的总量和交叉特征的分布也是相对固定的， 不会大批量出现新的term， 也不会出现交叉特征分布的大幅度变化。因此， 我们可以训练一个termDNN后持久化的在线上服务， 不必频繁的更新他。

但是IdDNN不一样， 首先， ID会大量出现新的（新用户）， 即使是旧用户， 由于用户的兴趣是在持续不断的变化中的，因此clientId\_X_Term类的feature的embedding的含义可能很不一样， 分布也会发生剧烈的变化， 在实践中， 我们会注意到，单次训练得到的模型， 在一段时间后AUC会发生剧烈下降， 因此，IdDNN模型我们需要至少daily的去更新他， 这样可以保证IdDNN的 auc 稳定在高点。

## 模型架构与训练

除了上述区别外， 模型在架构上大体一致， IdDNN可以训练的稍微深一些。 一般的， 我们使用300-150-64-1的模型架构， 在每一层都配上batchnorm和0.1的dropout即可。
激活函数选择relu， 带bias。这些被实验证明都是存在实际的帮助的。

模型训练时， 如前所述， TermDNN只需一次性训练即可， 而IdDNN需要daily train.

## 模型规模缩减——频率过滤算法

如前所述， 离散DNN模型可以非常大， 主要是最底层的embedingg层巨大， 模型总体的大小可能达到几十G, 但是很多场景下我们的资源有限， 不可能host这么大的模型。 因此， 我们需要一种算法， 在尽可能不降低AUC的情况下， 减少embedding 的数量， 从而显著减小模型规模。

我们采用特征值频率过滤算法。                                                                                                                                                                                         

```
usage =  last usage*x + score*(1-x)

score = impression * lx + click * ly

weighted_score = sigma(wx)

```

在所有embedding中，我们会见到大量的全0向量， 定性上理解， 可以认为当某个特征出现某个特征值时， 对最终的分类问题没有帮助，因此， 通过weighted_Score将score与w加权求和， 保证低权值和低展现的特征值均被过滤， 达到更好的效果。因此，通过以上公式， 我们可以将数百亿个feaure value按频率打分， 再取topN保存其离散层embedding， 在保证AUC没有显著损失的前提下， 有效降低模型规模。

模型缩减和daily train的关系：

值得一提的是， daily train时， 我们需要在离线记录和保存每一个特征值的embdding和last score， 然后在online service前将当天的model按如上算法进行cutting， 也就是说， daily train的整个过程中不涉及cut后的模型， cut模型仅在online serving的时候使用。

 



