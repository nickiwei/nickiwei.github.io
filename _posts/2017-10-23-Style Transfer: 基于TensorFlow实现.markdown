---
layout:     post
title: 风格迁移StyleTransfer(基于TensorFlow实现)
date:       2017-09-15 13:00:00
author:     "nickiwei"
header-img: "img/post-bg-2015.jpg"
tags:
    - 深度学习
---

*欢迎转载， 转载请注明出处及链接。*

*完整代码库请查看我的GithubRepo: <https://github.com/nick6918/MyDeepLearning> .部分代码参考了Stanford CS231n 课程作业。*

![style](/img/style.png)

如图，我们希望实现如下的系统， 使得任意输入一张content image和一张style image， 都能融合出一张新的style content image.

这被称为style transfer project, 它是Gredient Ascent的典型应用之一。（关于Gredient Ascent, 可以查看我的这篇博客）

如前所述， 在Gredient Ascent类问题中， 一个关键问题是， 合适的loss的选择。 在本例中， 我们使用三个不同的loss， 分别是style loss, content loss和total variation loss, 分别用来计算产生style feature, content feature的loss以及融合loss. 

PS, 我们将继续使用基于ImageNet训练的SqueezeNet作为base CNN, 它能够达到80%-85%的ImageNet正确识别率。

## Content Loss

Content Loss衡量content activation map与原始content image之间的差别， 如下：

![style](/img/contentLoss.png)

其中， Fl是第l层layer的activation map， p是source image的feature map， 我们计算二者差别的l2 loss, wc是一个decay标量。

实现如下：

```python
def content_loss(content_weight, content_current, content_original):
    """
    Compute the content loss for style transfer.
    
    Inputs:
    - content_weight: scalar constant we multiply the content_loss by.
    - content_current: features of the current image, Tensor with shape [1, height, width, channels]
    - content_target: features of the content image, Tensor with shape [1, height, width, channels]
    
    Returns:
    - scalar content loss
    """
    loss = content_weight * tf.reduce_sum((content_current - content_original)**2)
    return loss
```

很简单， 因为我们要计算feature之间的误差，所以直接计算了feature的L2 Loss.

## Style Loss

我们要如何衡量生成图片与style 图片之间在style上的差别呢？ 这是机器学习的一个很重要的思想， 使用一些数学参数来衡量我们想要衡量的指标， 从而产生一个可优化的Loss。

我们用图像feature map的feature neuron之间的相关性来代表图像的“风格”， 那么， 生成图片与style image之间的“风格”差异就可以用所有像素之间的相关性值的L2 Loss来表示。

如何表达所有像素两两之间的相关性呢， 这里， 我们用feature map的neuron之间的correlation相关系数来表达这种误差， 称之为gram map. 对于任意两个通道， 我们计算这两个通道之间每个元素的乘积(C*C)， 这代表了这两个通道之间的相关性， 共有H\*W这么多个这样的C\*C矩阵， 求这些矩阵的均值， 就代表了两张图所有通道之间的相关性， 称之为gram map.

![style](/img/styleloss.png)

综上， 我们经过两步推理， 得到了使用gram map的L2 loss来代表图像之间风格差异。
实现如下：

```python
def gram_matrix(features, normalize=True):
    """
    Compute the Gram matrix from features.
    
    Inputs:
    - features: Tensor of shape (1, H, W, C) giving features for
      a single image.
    - normalize: optional, whether to normalize the Gram matrix
        If True, divide the Gram matrix by the number of neurons (H * W * C)
    
    Returns:
    - gram: Tensor of shape (C, C) giving the (optionally normalized)
      Gram matrices for the input image.
    """
    shape = tf.shape(features)
    features_reshaped = tf.reshape(features, (shape[1]*shape[2], shape[3]))
    gram = tf.matmul(tf.transpose(features_reshaped), features_reshaped)
    if normalize:
        gram /= tf.cast((shape[3] * shape[1] * shape[2]), tf.float32)
    return gram


def style_loss(feats, style_layers, style_targets, style_weights):
    """
    Computes the style loss at a set of layers.
    
    Inputs:
    - feats: list of the features at every layer of the current image, as produced by
      the extract_features function.
    - style_layers: List of layer indices into feats giving the layers to include in the
      style loss.
    - style_targets: List of the same length as style_layers, where style_targets[i] is
      a Tensor giving the Gram matrix the source style image computed at
      layer style_layers[i].
    - style_weights: List of the same length as style_layers, where style_weights[i]
      is a scalar giving the weight for the style loss at layer style_layers[i].
      
    Returns:
    - style_loss: A Tensor contataining the scalar style loss.
    """
    total_loss = 0.0
    for i in range(len(style_layers)):
        G = style_targets[i]
        A = gram_matrix(feats[style_layers[i]])
        total_loss += style_weights[i]*tf.reduce_sum((G - A)**2)

    return total_loss
```
注意， 这里在计算gram map时， 有一个vectorize的小技巧。请自行理解。

## Total Variation Loss
有了Style Loss 和 Content Loss， 分别代表生成图片与style 图片之间的“风格差异”， 以及生成图片与content 图片之间的“feature差异”。 理论上， 我们就可以得到一个融合了风格图片风格的内容图片。

但这里存在过拟合。我们需要加入regularization项， 如何理解这种regularization呢？ 我们认为， 过度拟合就是图像为了拟合风格/内容， 而在图片的相邻像素之间， 产生了巨大的差异。 使得图片“看起来不自然”， 因此我们直接用相邻像素之间的差做regularization项。

![style](/img/regloss.png)

实现如下：

```python
def tv_loss(img, tv_weight):
    """
    Compute total variation loss.
    
    Inputs:
    - img: Tensor of shape (1, H, W, 3) holding an input image.
    - tv_weight: Scalar giving the weight w_t to use for the TV loss.
    
    Returns:
    - loss: Tensor holding a scalar giving the total variation loss
      for img weighted by tv_weight.
    """
    # Your implementation should be vectorized and not require any loops!
    left_loss = tf.reduce_sum((img[:, 1:, :, :] - img[:, :-1, :, :])**2)
    down_loss = tf.reduce_sum((img[:, :, 1:, :] - img[:, :, :-1, :])**2)
    loss = tv_weight*(left_loss + down_loss)
    return loss
```

## Loss In Graph
有了之前的各种loss算法， 我们就可以代入SqueezeNet中， 去计算当前CG图的total loss.

```python
def style_transfer_loss(content_image, style_image, image_size, style_size, content_layer, content_weight,
                   style_layers, style_weights, tv_weight, init_random):
    """
    
    Inputs:
    - content_image: filename of content image
    - style_image: filename of style image
    - image_size: size of smallest image dimension (used for content loss and generated image)
    - style_size: size of smallest style image dimension
    - content_layer: layer to use for content loss
    - content_weight: weighting on content loss
    - style_layers: list of layers to use for style loss
    - style_weights: list of weights to use for each layer in style_layers
    - tv_weight: weight of total variation regularization term
    - init_random: initialize the starting image to uniform random noise
    """
    # Extract features from the content image
    content_img = preprocess_image(load_image(content_image, size=image_size))
    feats = model.extract_features(model.image)
    content_target = sess.run(feats[content_layer],
                              {model.image: content_img[None]})

    # Initialize generated image to content image

    if init_random:
        img_var = tf.Variable(tf.random_uniform(content_img[None].shape, 0, 1), name="image")
    else:
        img_var = tf.Variable(content_img[None], name="image")

    # Extract features from the style image
    style_img = preprocess_image(load_image(style_image, size=style_size))
    style_feat_vars = [feats[idx] for idx in style_layers]
    style_target_vars = []
    # Compute list of TensorFlow Gram matrices
    for style_feat_var in style_feat_vars:
        style_target_vars.append(gram_matrix(style_feat_var))
    # Compute list of NumPy Gram matrices by evaluating the TensorFlow graph on the style image
    style_targets = sess.run(style_target_vars, {model.image: style_img[None]})

    # Extract features on generated image
    feats = model.extract_features(img_var)
    # Compute loss
    c_loss = content_loss(content_weight, feats[content_layer], content_target)
    s_loss = style_loss(feats, style_layers, style_targets, style_weights)
    t_loss = tv_loss(img_var, tv_weight)
    loss = c_loss + s_loss + t_loss

    return loss, img_var, content_img, style_img
```

我们在算法中加入了一个init_random变量， 一般的， 为了改变content image的style， 我们直接将content image设置为初始image即可， 但我们也可以使用一个随机图像， 然后将style loss 或 content loss的weight均设为0， 以此， 我们希望恢复原图的feature or style， 这被称为是feature inversion. 随后， 我们会详细分析feature inversion的效果， 这里为了实现style transfer， 我们只需将该变量设为False即可。

## Style Transfer的实现

### 梯度下降

有了loss， 下一步我们思考的问题是如何去优化这个loss， 与之前Image Fooling和Deep Dream不同的是， 这次我们想尽可能减小这个loss， 因此， 不是梯度上升， 而是梯度下降。

我们采用Adam法来做梯度下降。

```python
with tf.variable_scope("optimizer") as opt_scope:
        train_op = tf.train.AdamOptimizer(lr_var).minimize(loss, var_list=[img_var])
```

之前我们遇到的算法， 都是直接minimize(loss), 这里加入了一个新参数var_list, 为此， 我们查看一下minimize的API.

```
tf.train.Optimizer.minimize(loss, global_step=None, var_list=None, gate_gradients=1, name=None)

Add operations to minimize 'loss' by updating 'var_list'.

This method simply combines calls compute_gradients() and apply_gradients(). If you want to process the gradient before applying them call compute_gradients() and apply_gradients() explicitly instead of using this function.

Args:

loss: A Tensor containing the value to minimize.
global_step: Optional Variable to increment by one after the variables have been updated.
var_list: Optional list of variables.Variable to update to minimize 'loss'. Defaults to the list of variables collected in the graph under the key GraphKeys.TRAINABLE_VARIABLES.
gate_gradients: How to gate the computation of gradients. Can be GATE_NONE, GATE_OP, or GATE_GRAPH.
name: Optional name for the returned operation.
Returns:

An Operation that updates the variables in 'var_list'. If 'global_step' was not None, that operation also increments global_step.
```

可见， 我们通过对var\_list里的参数进行基于LOSS的梯度更新， 若var\_list为None, 则查看原CG图中属于GraphKeys.TRAINABLE_VARIABLES的所有变量。

当我们声明一个新的tensor变量时， 如果trainable=True(默认True), 则该变量会自动加入到GraphKeys.TRAINABLE_VARIABLES中。

### 实现

明确了梯度下降的策略后， 我们便可实现训练算法， 返回最终图像的同时， 我们还可以根据需要， 返回一些中间图像。

```python
def style_transfer_train(loss, img_var, initial_lr=3.0, decayed_lr=0.1, decay_lr_at=180, max_iter=200, print_every=50):
    # Create and initialize the Adam optimizer
    lr_var = tf.Variable(initial_lr, name="lr")
    # Create train_op that updates the generated image when run
    with tf.variable_scope("optimizer") as opt_scope:
        train_op = tf.train.AdamOptimizer(lr_var).minimize(loss, var_list=[img_var])
    # Initialize the generated image and optimization variables
    opt_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=opt_scope.name)
    sess.run(tf.variables_initializer([lr_var, img_var] + opt_vars))
    # Create an op that will clamp the image values when run
    clamp_image_op = tf.assign(img_var, tf.clip_by_value(img_var, -1.5, 1.5))

    imgs_in_process = []

    # Hardcoded handcrafted 
    for t in range(max_iter):
        # Take an optimization step to update img_var
        sess.run(train_op)
        if t < decay_lr_at:
            sess.run(clamp_image_op)
        if t == decay_lr_at:
            sess.run(tf.assign(lr_var, decayed_lr))
        if t % print_every == 0:
            print("train step: %d" % t)
            img = sess.run(img_var)
            imgs_in_process.append(img[0])
    print("train step: %d" % t)
    final_img = sess.run(img_var)[0]
    return imgs_in_process, final_img
```

## 测试

设置各种参数。

```python
# Set up optimization hyperparameters
initial_lr = 3.0
decayed_lr = 0.1
decay_lr_at = 180
max_iter = 200
print_every = 50
# Composition VII + Tubingen
params1 = {
    'content_image' : 'styles/tubingen.jpg',
    'style_image' : 'styles/composition_vii.jpg',
    'image_size' : 192,
    'style_size' : 512,
    'content_layer' : 3,
    'content_weight' : 5e-2, 
    'style_layers' : (1, 4, 6, 7),
    'style_weights' : (20000, 500, 12, 1),
    'tv_weight' : 5e-2,
    'init_random' : False
}
```

运行代码， 将所有图像显示出来。

```python
loss, img_var, content_img, style_img = style_transfer_loss(**params1)
imgs_in_process, final_img = style_transfer_train(loss, img_var)

#show original image
f, axarr = plt.subplots(1,2)
axarr[0].axis('off')
axarr[1].axis('off')
axarr[0].set_title('Content Source Img.')
axarr[1].set_title('Style Source Img.')
axarr[0].imshow(deprocess_image(content_img))
axarr[1].imshow(deprocess_image(style_img))
plt.show()

#show image in process
f, axarr = plt.subplots(1,len(imgs_in_process))
for i in range(len(imgs_in_process)):
    current_step = i * print_every + 1
    current_img = imgs_in_process[i]
    axarr[i].axis='off'
    axarr.set_title("Iteration %d" % current_step)
    axarr[i].imshow(deprocess_image(current_img, rescale=True))
plt.show()

#show final result
current_step = max_iter - 1
axarr.set_title("Final result at itertion %d" % current_step)
current_img = final_img
plt.imshow(deprocess_image(final_img, rescale=True))
plt.axis('off')
```

结果如下(在设置中更换各种图片和风格)：

###<i>Composition VII + Tubingen</i>
### 原始图像
![source1](/img/source1.png)
### 中间图像
![source1](/img/process1.png)
### 最终结果
![source1](/img/result1.png)

我们来尝试一下经典的starrynight风格的golden gate 金门大桥。

![source3](/img/GoldenGate.jpeg)

![source1](/img/result2.png)

可以看到， 虽然有了starry_night卷曲笔触的风格， 但不够锋利， 可以试着降低regularization, 另外， 颜色上相似度也不高， 可以试着提高style loss的比重， 然后增加训练次数。 

更多的调优这里就不赘述了， 最终的目标效果可见文首， 具体参数在论文中也有记录。 请大家自行查阅。

## 总结

style transfer是Gredient Ascent的一个经典应用， 结合上一篇文章中的Image Fooling和Deep Dream，包括基于RNN的image captioning， 甚至gan等， 他们都遵循了深度学习问题的基本范式。

关于深度学习的基本范式， 可以查看这篇博客<http://nickiwei.github.io/2017/09/16/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E5%BA%94%E7%94%A8%E9%97%AE%E9%A2%98%E7%9A%84%E5%9F%BA%E6%9C%AC%E8%8C%83%E5%BC%8F/>。

---

## 快速联系作者

欢迎关注我的知乎: <https://www.zhihu.com/people/NickWey> 


或直接在Github上联系我: <https://github.com/nick6918>
