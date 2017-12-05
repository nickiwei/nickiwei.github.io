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

接下来，我们尝试训练上一节中的浅层模型识别MNIST数据集。

## 构建trainer

首先， 我们利用已经写好的model构建trainer.

```python
def initialize_trainer():
	paddle.init(use_gpu=False, trainer_count=1)

	images = paddle.layer.data(
	    name='pixel', type=paddle.data_type.dense_vector(784))
	label = paddle.layer.data(
	    name='label', type=paddle.data_type.integer_value(10))

	# predict = softmax_classifer(images)
	# predict = fcn(images) # uncomment for MLP
	predict = cnn(images) # uncomment for LeNet5

	#cross entrophy classification cost
	cost = paddle.layer.classification_cost(input=predict, label=label)

	parameters = paddle.parameters.create(cost)

	optimizer = paddle.optimizer.Momentum(
	    learning_rate=0.1 / 128.0,
	    momentum=0.9,
	    regularization=paddle.optimizer.L2Regularization(rate=0.0005 * 128))

	trainer = paddle.trainer.SGD(cost=cost,
	                             parameters=parameters,
	                             update_equation=optimizer)

	return trainer
```

首先， 与Tensorflow中的placeholder概念类似， 在paddle中， 我们需要为数据构建data layer， 以方便后续进行处理。

```python
images = paddle.layer.data(
    name='pixel', type=paddle.data_type.dense_vector(784))
label = paddle.layer.data(
    name='label', type=paddle.data_type.integer_value(10))
```

完成后， 我们逐步完成了初始化parameters, 设置optimizer并最终生成一个trainer.

## 训练模型

构造trainer完成后， 我们就可以用所构造的trainer来训练我们的model了。 

```python
trainer = initialize_trainer()

# Train the model now
trainer.train(
    reader=paddle.batch(
        paddle.reader.shuffle(
            paddle.dataset.mnist.train(), buf_size=8192),
        batch_size=128),
    event_handler=event_handler,
    num_passes=5)
    
# find the best pass
best = sorted(lists, key=lambda list: float(list[1]))[0]
print 'Best pass is %s, testing Avgcost is %s' % (best[0], best[1])
print 'The classification accuracy is %.2f%%' % (100 - float(best[2]) * 100)
```

注意，在paddle中引入了异步event_handler机制， 我们可以由此， 在训练的特定阶段完成打印等任务。

这里， 我们构造一个event, 将每个阶段训练的loss和accuracy计算并打印出来。

```python
lists = []

#event handler to print the progress
def event_handler(event):
    if isinstance(event, paddle.event.EndIteration):
        if event.batch_id % 100 == 0:
            print "Pass %d, Batch %d, Cost %f, %s" % (
                event.pass_id, event.batch_id, event.cost, event.metrics)
    if isinstance(event, paddle.event.EndPass):
        # save parameters
        with open('params_pass_%d.tar' % event.pass_id, 'w') as f:
            trainer.save_parameter_to_tar(f)

        result = trainer.test(reader=paddle.batch(
            paddle.dataset.mnist.test(), batch_size=128))
        print "Test with Pass %d, Cost %f, %s\n" % (
            event.pass_id, result.cost, result.metrics)
        lists.append((event.pass_id, result.cost,
                      result.metrics['classification_error_evaluator']))
```

在event_handler的设计中， 我们通常需要针对不同的状态作出不同的操作， 因此，我们需要检查event的状态类型， 在paddle中， 一共给出了六种基本的event类型， 具体如下:
![event](/Users/weifanding/Desktop/picunuploaded/event.jpg)

## 测试模型

完成模型的训练后， 我们进行简单的效果测试。 由于MNIST数据集非常简单， 所以即使是浅层模型， 经过简单的训练， 也能得到很好的效果。

```python
def load_image(file):
    im = Image.open(file).convert('L')
    im = im.resize((28, 28), Image.ANTIALIAS)
    im = np.array(im).astype(np.float32).flatten()
    im = im / 255.0
    return im

test_data = []
cur_dir = os.getcwd()
test_data.append((load_image(cur_dir + '/image/infer_3.png'),))

probs = paddle.infer(
    output_layer=predict, parameters=parameters, input=test_data)
lab = np.argsort(-probs) # probs and lab are the results of one batch data
print "Label of image/infer_3.png is: %d" % lab[0][0]
```
在paddle中, 我们使用paddle.infer()接口测试模型。 注意， 在trainer中也有一个test方法， 该方法使用训练数据进行测试，得到的是训练集的acurracy.  

## 延伸阅读

下一节我们将开始构造一些经典的深度网络， 并利用其进行更加复杂的任务学习。


# 快速联系作者

欢迎关注我的知乎: https://www.zhihu.com/people/NickWey

或直接在Github上联系我: https://github.com/nick6918
