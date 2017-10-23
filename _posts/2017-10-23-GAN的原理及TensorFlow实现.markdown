---
layout:     post
title: GAN的原理及TensorFlow实现
date:       2017-09-25 12:01:00
author:     "nickiwei"
header-img: "img/post-bg-2015.jpg"
tags:
    - 深度学习
    - TensorFlow
    - GAN
---

*欢迎转载， 转载请注明出处及链接。*

*完整代码库请查看我的GithubRepo: <https://github.com/nick6918/MyDeepLearning> .部分代码参考了Stanford CS231n 课程作业。*


## GAN的原理

在Generative Model领域，一直存在两种流派， 一种是Explicit Density,指有明确的分布概率函数， 模型的目标是实现或逼近这种分布， 典型的例子是Enconder-Decoder模型和pixelRNN/pixelCNN模型。这二种模型的问题均在于performance和accuracy不高， 直到2014年， 非监督领域同样开始引入用神经网络代替复杂的函数or分布， 出现了GAN, 生成模型的performance得以提高。

GAN的基本原理来自于我们之前提到的Image Fooling思想， 我可以通过梯度上升生成一个图像， 使得它足够接近某一class的图像， 以至于分类器不能区分它和其他label为该类型的训练图像。

于是， 我们构造一个这样的Generator， 专门生成某一分类的图像。 同时， 为了提高他的精确度， 我们再构造一个专门识别这类图像的discriminator， 我们的目标是：

```
训练阶段：

1， 通过训练集训练Discrimator准确识别某一label的图像。

2， 提高梯度上升使得Generator尽可能产生让Dsicriminator“认为”是该label的图像。
```

为此，我们遵循深度学习问题的一般范式（具体请查看我的这篇博客），首先，构造如下Loss:
![loss](/img/LOSS3.png)

其次， 为该Loss选择合适的梯度上升or下降策略：

![loss](/img/loss.png)

如上， 我们希望尽可能提高Discriminator的准确率， 同时， Generator尽可能生成更准确的图像， 因此， 构造了这样的最大最小问题。

将最大最小问题转换为梯度策略：

![loss](/img/gre1.png)

将max转换为梯度上升， min转换为梯度下降， 此时， 我们发现一个问题， 即， 在训练开始阶段， 梯度较小， 模型收敛慢， 在训练结束阶段， 梯度较大， 模型难于收敛。
因此， 我们作如下变换：


![loss](/img/gre2.png)
如上， 在保持最大问题不变的前提下， 将最小问题转换为最大问题。此时， 梯度变为更适合训练的下凹函数。

## DCGAN的实现

为了提高GAN的performance， 我们引入共享参数， 使用卷积网络来拟合分布。

### GAN Loss

用cross entropy loss 来代替log(D(x)). D(x)为discriminator的输出结果。

```python
def gan_loss(logits_real, logits_fake):
    """Compute the GAN loss."""

    D_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = tf.ones_like(logits_real), logits=logits_real)) +  tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = tf.zeros_like(logits_fake), logits=logits_fake))
    G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = tf.ones_like(logits_fake), logits=logits_fake))
    return D_loss, G_loss  
```

### Discriminator

```python
def discriminator(x):
    """Compute discriminator score for a batch of input images.
    
    Inputs:
    - x: TensorFlow Tensor of flattened input images, shape [batch_size, 784]
    
    Returns:
    TensorFlow Tensor with shape [batch_size, 1], containing the score 
    for an image being real for each input image.
    """

    with tf.variable_scope("discriminator"):
        # TODO: implement architecture
        X_reshaped = tf.reshape(x, shape=[-1, 28, 28, 1])
        H1 = tf.layers.conv2d(inputs=X_reshaped, filters=32, kernel_size=5, strides=1,activation=None, padding='VALID', use_bias=True)
        H1D = leaky_relu(H1)
        H1_pooled = tf.layers.max_pooling2d(inputs = H1D, strides=2, pool_size=2)
        H2 = tf.layers.conv2d(inputs=H1_pooled, filters=64, kernel_size=5, strides=1,activation=None, padding='VALID', use_bias=True)
        H2D = leaky_relu(H2)
        H3 = tf.layers.max_pooling2d(inputs = H2D, strides=2, pool_size=2)
        H3_flattened = tf.reshape(H3, shape=[-1, 4*4*64])
        H4 = tf.layers.dense(inputs=H3_flattened, units=4*4*64, activation = None, use_bias = True)
        H4D = leaky_relu(H4)
        logits = tf.layers.dense(inputs=H4D, units=1, activation = None, use_bias = True)
        return logits
```
本质上就是一个CNN， 经过若干卷积层后最后用FCN输出（N, 1)的scores.

### Generator

```python
def generator(z):
    """Generate images from a random noise vector.
    
    Inputs:
    - z: TensorFlow Tensor of random noise with shape [batch_size, noise_dim]
    
    Returns:
    TensorFlow Tensor of generated images, with shape [batch_size, 784].
    """
    with tf.variable_scope("generator"):
        # TODO: implement architecture
        H1 = tf.layers.dense(inputs = z, units = 1024, activation = tf.nn.relu, use_bias = True)
        H1_BN = tf.layers.batch_normalization(inputs=H1, axis=1)
        H2 = tf.layers.dense(inputs = H1_BN, units = 7*7*128, activation = tf.nn.relu, use_bias = True)
        H2_BN = tf.layers.batch_normalization(inputs=H2, axis=1)
        H2_reshaped = tf.reshape(H2_BN, shape = [-1, 7, 7, 128])
        H3 = tf.layers.conv2d_transpose(inputs = H2_reshaped, strides = 2, filters = 64, kernel_size = 4, padding = 'SAME', activation =tf.nn.relu, use_bias = True)
        H3_BN = tf.layers.batch_normalization(inputs = H3, axis = 3)
        img = tf.layers.conv2d_transpose(inputs = H3_BN, strides = 2, filters = 1, kernel_size = 4, padding = 'SAME', activation =tf.nn.tanh, use_bias = True)
        return img
```
比较有技巧性的是generator的构造，我们从random noise出发， 首先通过FCN整型成目标图像的样子， 然后通过两个解卷积得到最终的img. 解卷积本质上仍是一个卷积， 但是卷积过程与原卷积恰好相反（包含了一个转置的过程）， 至于为什么要转置， 这里不详述了， 简单来说， 在padding后的原输入贡献给输出的位置恰好发生了一次转置。


有了discriminator和generator我们就可以来构造整个CG了。

### CG

```python
def construct_GAN_CG():

    tf.reset_default_graph()

    # number of images for each batch
    batch_size = 128
    # our noise dimension
    noise_dim = 96

    # placeholder for images from the training dataset
    x = tf.placeholder(tf.float32, [None, 784])
    # random noise fed into our generator
    z = sample_noise(batch_size, noise_dim)
    # generated images
    G_sample = generator(z)

    with tf.variable_scope("") as scope:
        #scale images to be -1 to 1
        logits_real = discriminator(preprocess_img(x))
        # Re-use discriminator weights on new inputs
        scope.reuse_variables()
        logits_fake = discriminator(G_sample)

    # Get the list of variables for the discriminator and generator
    D_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'discriminator')
    G_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'generator') 

    # get our solver
    D_solver, G_solver = get_solvers()

    # get our loss
    D_loss, G_loss = lsgan_loss(logits_real, logits_fake)

    # setup training steps
    D_train_step = D_solver.minimize(D_loss, var_list=D_vars)
    G_train_step = G_solver.minimize(G_loss, var_list=G_vars)
    D_extra_step = tf.get_collection(tf.GraphKeys.UPDATE_OPS, 'discriminator')
    G_extra_step = tf.get_collection(tf.GraphKeys.UPDATE_OPS, 'generator')
    return x, G_sample, G_train_step, G_loss, D_train_step, D_loss, G_extra_step, D_extra_step
```

## GAN的测试
我们使用如下代码训练一个GAN.

```python
def gan_train(sess, x, G_sample, G_train_step, G_loss, D_train_step, D_loss, G_extra_step, D_extra_step, show_every=250, print_every=50, batch_size=128, num_epoch=3, dataset=mnist):
    """Train a GAN for a certain number of epochs.
    
    Inputs:
    - sess: A tf.Session that we want to use to run our data
    - G_train_step: A training step for the Generator
    - G_loss: Generator loss
    - D_train_step: A training step for the Generator
    - D_loss: Discriminator loss
    - G_extra_step: A collection of tf.GraphKeys.UPDATE_OPS for generator
    - D_extra_step: A collection of tf.GraphKeys.UPDATE_OPS for discriminator
    Returns:
        Nothing
    """
    # compute the number of iterations we need
    max_iter = int(dataset.train.num_examples*num_epoch/batch_size)

    imgs_in_process = []

    for it in range(max_iter):
        # every show often, show a sample result
        if it % show_every == 0:
            samples = sess.run(G_sample)
            # fig = show_images(samples[:16])
            # plt.show()
            imgs_in_process.append(samples[:16])
            print("Saved images in iter %d" % it)
        # run a batch of data through the network
        minibatch, minbatch_y = dataset.train.next_batch(batch_size)
        _, D_loss_curr = sess.run([D_train_step, D_loss], feed_dict={x: minibatch})
        _, G_loss_curr = sess.run([G_train_step, G_loss])

        # print loss every so often.
        # We want to make sure D_loss doesn't go to 0
        if it % print_every == 0:
            print('Iter: {}, D: {:.4}, G:{:.4}'.format(it,D_loss_curr,G_loss_curr))
    return imgs_in_process, G_sample
```
我们打印出一些训练中间的图片。

![loss](/img/mnist1.png)

训练完成后， 尝试利用generator输出一些结果， 如下：

![loss](/img/mnist2.png)

可见， 最终的输出结果已经非常逼近mnist的输入了。

ps, GAN的训练依赖于GPU, 由于设备限制， 我在CPU上完成了训练。
基本上， 3个epoch大约2000个iteration的训练， 每个iteration完成200个训练集的minibatch， 在CPU上耗时达到10分钟以上， 10个epoch则需要半小时以上。有条件的同学可以尽量使用GPU来训练。

## GAN的优化

传统的DCGAN的问题是难于converge， 为此， 产生了一系列的优化型GAN, 这些GAN都从某一方面帮助GAN进行收敛。

### LSGAN

LSGAN使用Least Square Loss代替sigmoid loss， 实证证明具有更好的收敛性。

![LSLoss](/img/lsloss.png)


```python
def lsgan_loss(score_real, score_fake):
    """Compute the Least Squares GAN loss.
    
    Inputs:
    - score_real: Tensor, shape [batch_size, 1], output of discriminator
        score for each real image
    - score_fake: Tensor, shape[batch_size, 1], output of discriminator
        score for each fake image    
          
    Returns:
    - D_loss: discriminator loss scalar
    - G_loss: generator loss scalar
    """
    # TODO: compute D_loss and G_loss
    G_loss = 0.5 * tf.reduce_mean((score_fake-1)**2)
    D_loss = 0.5 * tf.reduce_mean((score_real-1)**2)\
             + 0.5 * tf.reduce_mean(score_fake**2)
    return D_loss, G_loss
```

除此之外，AFI, BiGAN, Softmax GAN, Conditional GAN, InfoGAN都属于此类改变loss function以提高GAN converge能力的优化。读者可自行研究。 此类型中最重要的优化就是WGAN了， 关于WGAN， 我们会在下一篇博文中详述。

### 半监督学习

除此之外，对于部分label数据， 我们可以采用半监督的方式优化GAN, 这部分我会再专门写一篇博客， 读者也可自行参考该论文（https://arxiv.org/pdf/1606.03498.pdf ）。

## 结语

作为目前非监督学习中performance最好的一种， GAN放弃了pixelCNN/RNN 以及VAE等模型所追求的explicit density function, 改为了用对抗网络去拟合一个分布， 他讲一个随机的高斯分布作为输入， 输出一个目标label的生成分布。

但普通的DC-GAN仍然存在难于收敛， 难于调优等问题， 为此， 人们设计多种多样的优化GAN， 包括 InfoGAN等， 这其中performance最好的WGAN， 我们将在下一篇文章中详述。
