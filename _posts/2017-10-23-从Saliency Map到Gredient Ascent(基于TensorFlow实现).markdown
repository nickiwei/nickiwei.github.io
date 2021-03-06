---
layout:     post
title: 从Saliency Map到Gredient Ascent(基于TensorFlow实现)
date:       2017-09-19 12:01:00
author:     "nickiwei"
header-img: "img/post-bg-2015.jpg"
tags:
    - 深度学习
    - TensorFlow
    - DeepDream
---

*欢迎转载， 转载请注明出处及链接。*

*完整代码库请查看我的GithubRepo: <https://github.com/nick6918/MyDeepLearning> .部分代码参考了Stanford CS231n 课程作业。*

在训练模型时， 我们往往关注于Loss对各种模型的待估参数(W, b, gamma, beta等)的梯度， 从而优化这些参数以达到更好的训练效果。

在本文中， 我们则转换视角， 关注Loss对输入图像(pixel)的梯度， 从而利用这些梯度完成图像分析， 图像生成等工作，并着重分析几个非常经典的工程案例, Image Fooling, Deep Dream和style transfer, 其核心思想均是Gredient Ascent.

# Saliency Map(显著性图)

Saliency Map的关键idea是， 通过计算梯度， 反映出图像的哪一部分对分类的作用最大。

## 实现
直接上代码

```python
def compute_saliency_maps(X, y, model):
    """
    Compute a class saliency map using the model for images X and labels y.

    Input:
    - X: Input images, numpy array of shape (N, H, W, 3)
    - y: Labels for X, numpy of shape (N,)
    - model: A SqueezeNet model that will be used to compute the saliency map.

    Returns:
    - saliency: A numpy array of shape (N, H, W) giving the saliency maps for the
    input images.
    """
    correct_scores = tf.gather_nd(model.classifier, tf.stack((tf.range(X.shape[0]), model.labels), axis=1))
    losses = tf.square(1 - correct_scores)
    grad_img = tf.gradients(losses ,model.image)
    grad_img_val = sess.run(grad_img,feed_dict={model.image:X,model.labels:y})[0]
    saliency = np.sum(np.maximum(grad_img_val,0),axis=3)
    return saliency
```

这段code的每一句都值得细细分析， 

```python
correct_scores = tf.gather_nd(model.classifier, 
\tf.stack((tf.range(X.shape[0]), model.labels), axis=1))
```
其中，

tf.stack() 类似于np.hstack/vstack取决于axis.

tf.gather_nd(x, y) 类似于<ndarray>[x, y]， 其中x，y是两个一维向量。

	loss = tf.square(1 - correct_scores)
	
第二行， 计算loss， 这里loss的idea是， 结果不是正确的概率（由于数据经过normalization， 所以数据范围为0-1(最大值)

回顾SqueezeNet本身使用的loss

```python
self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(self.labels, NUM_CLASSES), logits=self.classifier))
```

可以看出， 他所求的是每个sample softmax loss的平均值。

	grad_img = tf.gradients(loss,model.image)
	grad_img_val = sess.run(grad_img,feed_dict={model.image:X,model.labels:y})[0]
	
第三四行， 这里显示的求取了dLoss\_dImg\_pixel

回顾在一般优化模型时， 我们是如何利用梯度优化模型的：

```python
variables = [loss_val,correct_prediction,accuracy]
    if training_now:
        variables[-1] = training
#...

loss, corr, _ = session.run(variables, feed_dict=feed_dict)

#...
optimizer = tf.train.AdamOptimizer(5e-4) # select optimizer and set learning rate

with tf.control_dependencies(extra_update_ops):
    training = optimizer.minimize(mean_loss)
           
```
总结一下， 在计算完loss后， 我们call training这个function， training function实际上等价于optimizer.minimize(mean_loss), 我们在这个function内部自动求梯度， 并完成梯度下降。

	saliency = np.sum(np.maximum(grad_img_val,0),axis=3)
	
最后一行， 我们求取了三个通道中的最大值。可见， 一句话总结的话， saliency map其实就是dLoss_dx在各通道的最大值。

注意， <b>在sess.run()之前， 我们进行的是函数式编程， 操作tensor，必须使用tf.xx方法； 在sess.run()之后， 我们进行的是带值的命令式编程， 操作的是numpy ndarray， 使用Numpy or Python方法</b>。

## Saliency Map的测试

我们使用如下的代码生成Saliency Map的结果

```python
def show_saliency_maps(X, y, mask):
    mask = np.asarray(mask)
    Xm = X[mask]
    ym = y[mask]
        
    saliency = compute_saliency_maps(Xm, ym, model)
    for i in range(mask.size):
        plt.subplot(2, mask.size, i + 1)
        plt.imshow(deprocess_image(Xm[i]))
        plt.axis('off')
        plt.title(class_names[ym[i]])
        plt.subplot(2, mask.size, mask.size + i + 1)
        plt.title(mask[i])
        plt.imshow(saliency[i], cmap=plt.cm.hot)
        plt.axis('off')
        plt.gcf().set_size_inches(10, 4)
    plt.show()

mask = np.arange(5)
show_saliency_maps(X, y, mask)
```

在ImageNet中，我们可以得到以下的结果：
 
![saliency](/img/slm.png)

可以看到， 大致上saliency map找到了object 所在的关键部分， 可以把saliency map当作是segmentation的一种非监督方法， 但精度和performance都不高就是了。

## Loss 比较

```python
correct_scores = tf.gather_nd(model.classifier, tf.stack((tf.range(X.shape[0]), model.labels), axis=1))
losses = tf.square(1 - correct_scores)

losses = tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(model.labels, model.classifier.shape[1]), logits=model.classifier)
```
在本例中， 我们直接使用了基于正确分类score的L2 Loss, 这个loss乍一看可能有一些困惑， 其实我们也可以使用一般的softmax cross entropy loss， 在替换loss后， 我们得到了如下结果

![saliency](/img/1211.png)

与原结果基本相似， 但仔细查看会注意到以下几点：

1， 使用L2 Loss的结果在核心区域的亮度更高， 意味着核心区域与非核心区域的梯度差别更大。 这才图0(hay)中很明显， 关键草堆部分saliency map明显更亮一些。

2， L2 Loss核心区域边界更紧凑且准确。 这可以从图2(Tibetan Mastiff)中看到， L2 loss saliency map狗的轮廓要更明显且准确。

之所以产生这种现象， 是因为cross entropy不仅考虑了正确分类， 也考虑了不正确分类之间的关系， 因此， 反倒弱化了最终结果的边界和准确度。（这里我的理解也不是很透彻， 欢迎大家指正）

除此之外， L2 loss在计算上也更简单些。



# 从Saliency Map到Gredient Ascent

## Saliency Map引发的Gredient Ascent思想

在saliency map中， 我们固定了model已经训练好的参数， 然后计算出当前模型下每个pixel对最终正确分类loss的贡献dLoss_dPixel, 将该梯度可视化为saliency map.

由此产生的自然联系是， 既然又了dLoss_dPixel， 我们能否通过这个梯度更新pixel的值， 从而产生更准确分类甚至是固定错误分类（选择错误分类来计算loss）的图片。

Deep Dream 和 Image Fooling 就是根据以上思想产生出的项目.

## Image Fooling

如前所述， 我们在一个已经训练好的分类器的基础上， 通过选择某一错误分类计算loss, 然后Gredient Ascent更新图片pixel， 使得分类器对图片错误分类。

## 实现

```python
def make_fooling_image(X, target_y, model):
    """
    Generate a fooling image that is close to X, but that the model classifies
    as target_y.

    Inputs:
    - X: Input image, of shape (1, 224, 224, 3)
    - target_y: An integer in the range [0, 1000)
    - model: Pretrained SqueezeNet model

    Returns:
    - X_fooling: An image that is close to X, but that is classifed as target_y
    by the model.
    """
    X_fooling = X.copy()
    learning_rate = 1
    
    for i in range(100):
    	scores = sess.run(model.classifier, feed_dict = {model.image: X_fooling})
    	print('step:%d,current_label_score:%f,target_label_score:%f' % \
              (i,scores[0].max(),scores[0][target_y]))
    	predict_y = np.argmax(scores[0])
    	if predict_y == target_y:
    		break
    	losses = scores[0, target_y]
    	
    	grad_img = tf.gradients(model.classifier[0,target_y],model.image)[0]
    	grad_img_val = sess.run(grad_img, feed_dict = {model.image: X_fooling})
    	grad_img_val = grad_img_val[0]
    	dX = learning_rate * grad_img_val / np.sum(grad_img_val)
    	X_fooling += dX
    return X_fooling
```

其核心思想在于将图片pixel沿着梯度方向改变， 使得

	dX = learning_rate * grad_img_val / np.sum(grad_img_val)
    X_fooling += dX

## 测试

我们使用如下的代码可视化Fooling Image

```python
def show_image_fooling(X, X_fooling):
	# Show original image, fooling image, and difference
	orig_img = deprocess_image(X[0])
	fool_img = deprocess_image(X_fooling[0])
	# Rescale 
	plt.subplot(1, 4, 1)
	plt.imshow(orig_img)
	plt.axis('off')
	plt.title(class_names[y[idx]])
	plt.subplot(1, 4, 2)
	plt.imshow(fool_img)
	plt.title(class_names[target_y])
	plt.axis('off')
	plt.subplot(1, 4, 3)
	plt.title('Difference')
	plt.imshow(deprocess_image((Xi-X_fooling)[0]))
	plt.axis('off')
	plt.subplot(1, 4, 4)
	plt.title('Magnified difference (10x)')
	plt.imshow(deprocess_image(10 * (Xi-X_fooling)[0]))
	plt.axis('off')
	plt.gcf().tight_layout()

	plt.show()

#select a pic and decide a fooling label
Xi = X[0][None]
target_y = 6
X_fooling = make_fooling_image(Xi, target_y, model)

show_image_fooling(Xi, X_fooling)
```
注意,

	Xi = X[0][None]
我们希望任选一张图进行Image Fooling, 但不能直接写 Xi = X[0], 因为此时， TF并不知道X的形状， 为了表示剩下的shape由灌进来的输入决定， 需改写为 Xi = X[0][None].

最终结果如下：

![imageFooling](/img/IMGfOOL.png)

## Deep Dream: Amplify existing features

与Image Fooling的思想恰好相反， DeepDream想要通过Gredient Ascent, 增强已有的feature， 来降低目标label的loss, 从而产生一幅增强feature的新图， 一方面， 所产生的新图对于目标label的确信度（score as possiblity）更高， 一方面， 我们可以从该图中大概看出， 系统到底在寻找怎样的feature.

Deep Dream 原始Project在这里： https://github.com/google/deepdream

## 实现
直接上代码

```python
def create_class_visualization(target_y, model, **kwargs):
    """
    Generate an image to maximize the score of target_y under a pretrained model.
    
    Inputs:
    - target_y: Integer in the range [0, 1000) giving the index of the class
    - model: A pretrained CNN that will be used to generate the image
    
    Keyword arguments:
    - l2_reg: Strength of L2 regularization on the image
    - learning_rate: How big of a step to take
    - num_iterations: How many iterations to use
    - blur_every: How often to blur the image as an implicit regularizer
    - max_jitter: How much to gjitter the image as an implicit regularizer
    - show_every: How often to show the intermediate result
    """
    l2_reg = kwargs.pop('l2_reg', 1e-3)
    learning_rate = kwargs.pop('learning_rate', 25)
    num_iterations = kwargs.pop('num_iterations', 100)
    blur_every = kwargs.pop('blur_every', 10)
    max_jitter = kwargs.pop('max_jitter', 16)
    show_every = kwargs.pop('show_every', 25)

    X = 255 * np.random.rand(224, 224, 3)
    X = preprocess_image(X)[None]

    losses = model.classifier[0] 
    grad = tf.gradients(model.classifier[0, target_y], model.image)[0] - l2_reg*model.image

    for t in range(num_iterations):
        # Randomly jitter the image a bit; this gives slightly nicer results
        ox, oy = np.random.randint(-max_jitter, max_jitter+1, 2)
        Xi = X.copy()
        X = np.roll(np.roll(X, ox, 1), oy, 2)
        
        loss_val = sess.run(losses, feed_dict={model.image: X})
        grad_val = sess.run(grad, feed_dict={model.image: X})
        dX = learning_rate * grad_val  
        X += dX

        print('step:%d,current_label_score:%f,target_label_score:%f' % \
              (t, loss_val.max(), loss_val[target_y]))

        # Undo the jitter
        X = np.roll(np.roll(X, -ox, 1), -oy, 2)

        # As a regularizer, clip and periodically blur
        X = np.clip(X, -SQUEEZENET_MEAN/SQUEEZENET_STD, (1.0 - SQUEEZENET_MEAN)/SQUEEZENET_STD)
        if t % blur_every == 0:
            X = blur_image(X, sigma=0.5)

        # Periodically show the image
        if t == 0 or (t + 1) % show_every == 0 or t == num_iterations - 1:
            plt.imshow(deprocess_image(X[0]))
            class_name = class_names[target_y]
            plt.title('%s\nIteration %d / %d' % (class_name, t + 1, num_iterations))
            plt.gcf().set_size_inches(4, 4)
            plt.axis('off')
            plt.show()
    return X
```

在代码中， 还包括了jitter， clip和peridically blur等技术， 其目的都是希望特征能更准确的显示出来， 在Google的项目日志中都有较详细的介绍， 这里不再细讲。

## 测试

```
target_y = 366 # Gorilla  
out = create_class_visualization(target_y,model,num_iterations=200)
```
任选一个类型， 开始测试， 结果如下：

![imageFoolingdata](/img/6.png)
![imageFooling1](/img/1.png)
![imageFooling2](/img/2.png)
![imageFooling3](/img/3.png)
![imageFooling4](/img/4.png)
![imageFooling5](/img/5.png)

可以看出， 在60多iteration左右， 目标label已经是score最高的label了， 从图像中， 我们可以隐约看出几个猩猩头。 在Google 博客中， 我们看一看到一些经过调优以后的最终效果，如下：

![imageFooling6](/img/ddper.png)

## 结语

本文从Salency Map出发， 介绍了Gredient Ascent算法及其在图像生成方面的两个典型应用， 从中， 我们可以对NN在feature 提取方面的效果有更生动的认识。 

### Loss Funtion

特别注意到， 为了满足不同目的的gredient ascent, 我们选择了不同的目标loss function,总结如下:

```python
#Saliency Map
#选择(1 - 正确score)的L2 loss
correct_scores = tf.gather_nd(model.classifier, tf.stack((tf.range(X.shape[0]), model.labels), axis=1))
losses = tf.square(1 - correct_scores) #(N, )
    
#  Image Fooling
# 目标label的score作为loss
scores = sess.run(model.classifier, feed_dict = {model.image: X_fooling})
losses = scores[0, target_y] #scalar

#Deep Dream
# 每个label的score均作为loss
losses = model.classifier[0] #(D, )
```

可见， 虽然都是在计算loss对image的每个pixel的gredients， 但由于目标不同， 我们所采用的loss也不同， 合理选择loss对于Gredient Ascent类问题至关重要。

除此之外， 这一方面目前最重要的应用之一是style transfer和基于style transfer的图片处理应用。我们将在下一篇博客中详细介绍。

---

## 快速联系作者

欢迎关注我的知乎: <https://www.zhihu.com/people/NickWey> 


或直接在Github上联系我: <https://github.com/nick6918>
