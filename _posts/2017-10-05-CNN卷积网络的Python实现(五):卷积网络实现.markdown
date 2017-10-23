---
layout:     post
title: CNN卷积网络的Python实现(五):卷积网络实现
date:       2017-09-15 13:00:00
author:     "nickiwei"
header-img: "img/post-bg-2015.jpg"
tags:
    - 深度学习
---

这个系列从最基础的全连接网络开始， 从零开始实现包含CNN, RNN等在内的深度网络模型。本文是该系列的第五篇， 介绍卷积网络的实现。

*欢迎转载， 转载请注明出处及链接。*

*完整代码库请查看我的GithubRepo: https://github.com/nick6918/MyDeepLearning .部分代码参考了Stanford CS231n 课程作业。*

常见的卷积网路如图所示
![architecture](/img/dlarch.jpeg)

本文将搭建一个简单的三层卷积网络， 然后实现一个基本的Solver. 在实际的框架中， 仅model及其tuning是需要手动实现的， 其余均已自动化，但这里，我们还是会手动实现一个solver 以及手动计算相应梯度。

## Solver的实现

我们将对model的training封装进一个叫Solver的对象中， 如图可见， solver的核心是利用optimizer算法对model的各个参数进行梯度下降优化。 同时， 我们会记录下一些中间数据以便可视化。

各种Optimizer算法（如SGD_Momentum, AdaGrad, Adam等）已在之前的CNN面面观系列中介绍过， 这里不再赘述。

以下是Solver的核心API:

```python
	_step()
	check_accuracy(self, X, y, num_samples=None, batch_size=100)
	train()
```

train()方法是核心的训练算法， train方法会call \_step()方法， 在_step()中， 我们进行一步梯度下降。

checkaccuracy()中， 我们将测试数据导入测试当前模型的训练成果。

读者可直接阅读代码库中的/lib/solvers/solver.py文件， 了解solver的实现。Link: <https://github.com/nick6918/MyDeepLearning/blob/master/lib/solvers/solver.py>

代码虽然较多， 但注释充分， 读者可自行理解一下。

注意事项：
1， 我们不直接设置iteration的次数， 而是引入epoch的概念， 所有sample被用来训练一遍， 称为一个epoch. 在每个iteration中， 我们导入一个bacthsize的sample用来训练。

综上， 对于N组数据， 完成i个epoch所需的iteration次数为:

	num_iter = i * N / batchsize
	
2, 在训练过程中， 我们会记录下每个epoch的training accuracy和validation accuracy, 还会记录下每个iteration的training loss以便之后分析使用。具体的使用案例见本文最后。

## Three Layer CNN的实现

### 基本架构

### First Layer: Conv Layer
Conv, BN, relu， max pool, dropout 

### Second Layer: FC Layer
affine, relu, dropout

### Third Layer: softmax Layer
affine, softmax

### Sandwich Layer
将一些常见的layer组合组合成一个新的layer供调用。这里仅举一例。

Forward Path

```python
def conv_relu_pool_forward(x, w, b, conv_param, pool_param):
    """
    Convenience layer that performs a convolution, a ReLU, and a pool.
    """
    a, conv_cache = conv_forward_fast(x, w, b, conv_param)
    s, relu_cache = relu_forward(a)
    out, pool_cache = max_pool_forward_fast(s, pool_param)
    cache = (conv_cache, relu_cache, pool_cache)
    return out, cache
```

Backward Path

```python
def conv_relu_pool_backward(dout, cache):
    """
    Backward pass for the conv-relu-pool convenience layer
    """
    conv_cache, relu_cache, pool_cache = cache
    ds = max_pool_backward_fast(dout, pool_cache)
    da = relu_backward(ds, relu_cache)
    dx, dw, db = conv_backward_fast(da, conv_cache)
    return dx, dw, db
```

### 实现
```python
class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:

    conv - bn - relu - 2x2 max pool - affine - bn - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
                 hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
                 dtype=np.float32, dropout=0, use_batchnorm=False, pool_param={'pool_height': 2, 'pool_width': 2, 'stride': 2}, conv_param=None):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Size of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype
        self.use_batchnorm = use_batchnorm
        self.dropout = dropout
        self.pool_param = pool_param
        if conv_param:
            self.conv_param = conv_param
        else:
            self.conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}
        if self.use_batchnorm:
            self.bn_param1 = {'mode': 'train'}
            self.bn_param2 = {'mode': 'train'}

        self.use_dropout = False
        if dropout > 0:
            self.use_dropout = True
            self.dropout_param = {'mode': 'train', 'p': dropout}

        C, H, W = input_dim
        Hn = (H - filter_size + 2 * self.conv_param['pad']) // self.conv_param['stride'] + 1
        Wn = (W - filter_size + 2 * self.conv_param['pad']) // self.conv_param['stride'] + 1
        Hp = (Hn - self.pool_param["pool_height"]) // self.pool_param["stride"] + 1
        Wp = (Wn - self.pool_param["pool_width"]) // self.pool_param["stride"] + 1
        
        # Conv layer initialization
        #(N, C, HW, WW)
        self.params["W1"] = weight_scale * np.random.randn(num_filters, C, filter_size, filter_size)
        self.params["b1"] = np.zeros(num_filters)
        # FC layer initialization
        self.params["W2"] = weight_scale * np.random.randn(num_filters * Hp * Wp, hidden_dim)
        self.params["b2"] = np.zeros(hidden_dim)
        #Loss layer initialization
        self.params["W3"] = weight_scale * np.random.randn(hidden_dim, num_classes)
        self.params["b3"] = np.zeros(num_classes)
        if self.use_batchnorm:
            #Spatial BN
            self.params["gamma1"] = np.ones(num_filters)
            self.params["beta1"] = np.zeros(num_filters)
            #Affine BN
            self.params["gamma2"] = np.ones(hidden_dim)
            self.params["beta2"] = np.zeros(hidden_dim)

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param['mode'] = mode
        if self.use_batchnorm:
            self.bn_param1['mode'] = mode
            self.bn_param2['mode'] = mode

        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']
        if self.use_batchnorm:
            gamma1, beta1 = self.params['gamma1'], self.params['beta1']
            gamma2, beta2 = self.params['gamma2'], self.params['beta2']

        # pass conv_param to the forward pass for the convolutional layer
        filter_size = W1.shape[2]
        #conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        #pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        caches = []
        if self.use_batchnorm:
            H1, cache = conv_bn_relu_pool_forward(X, W1, b1, gamma1, beta1, self.conv_param, self.bn_param1, self.pool_param)
            caches.append(cache)
            if self.use_dropout:
                H1, cache = dropout_forward(H1, self.dropout_param)
                caches.append(cache) 
            H1, cache = flatten_forward(H1)
            caches.append(cache)
            H2, cache = affine_bn_relu_forward(H1, W2, b2, gamma2, beta2, self.bn_param2)
            caches.append(cache)
            if self.use_dropout:
                H2, cache = dropout_forward(H2, self.dropout_param)
                caches.append(cache) 
        else:
            H1, cache = conv_relu_pool_forward(X, W1, b1, self.conv_param, self.pool_param)
            caches.append(cache)
            if self.use_dropout:
                H1, cache = dropout_forward(H1, self.dropout_param)
                caches.append(cache)
            H1, cache = flatten_forward(H1)
            caches.append(cache)
            H2, cache = affine_relu_forward(H1, W2, b2)
            caches.append(cache)
            if self.use_dropout:
                H2, cache = dropout_forward(H2, self.dropout_param)
                caches.append(cache)
        scores, cache = affine_forward(H2, W3, b3)
        caches.append(cache)
    
        if y is None:
            return scores

        loss, grads = 0, {}
        loss, dLoss_dscores = softmax_loss(scores, y)
        loss += 0.5*self.reg*np.sum(self.params["W3"]**2)
        loss += 0.5*self.reg*np.sum(self.params["W2"]**2)
        loss += 0.5*self.reg*np.sum(self.params["W1"]**2)
        dLoss_dH2, grads["W3"], grads["b3"] = affine_backward(dLoss_dscores, caches.pop())
        if self.use_dropout:
                dLoss_dH2 = dropout_backward(dLoss_dH2, caches.pop())
        if self.use_batchnorm:
            dLoss_dH1, grads["W2"], grads["b2"], grads["gamma2"], grads["beta2"] = affine_bn_relu_backward(dLoss_dH2, caches.pop())
            dLoss_dH1 = flatten_backward(dLoss_dH1, caches.pop())
            if self.use_dropout:
                dLoss_dH1 = dropout_backward(dLoss_dH1, caches.pop())
            dLoss_dX, grads["W1"], grads["b1"], grads["gamma1"], grads["beta1"] = conv_bn_relu_pool_backward(dLoss_dH1, caches.pop())
        else:
            dLoss_dH1, grads["W2"], grads["b2"] = affine_relu_backward(dLoss_dH2, caches.pop())
            dLoss_dH1 = flatten_backward(dLoss_dH1, caches.pop())
            if self.use_dropout:
                dLoss_dH1 = dropout_backward(dLoss_dH1, caches.pop())
            dLoss_dX, grads["W1"], grads["b1"] = conv_relu_pool_backward(dLoss_dH1, caches.pop())
        grads["W3"] += self.reg*self.params["W3"]
        grads["W2"] += self.reg*self.params["W2"]
        grads["W1"] += self.reg*self.params["W1"]

        return loss, grads

```

读者可自行理解一下上述代码。 大致上， 在init函数初始化各种待估变量， 在loss函数中构造CG(computational graph及其反向传播)， 下面我们重点介绍一下， 如何测试模型的正确性。

## 检查模型的正确性

### 方案一， 计算初始loss

对随机数据的初识loss有一个大致的判别， 如果初识loss过大(一般不会超过10)， 则需检查实现。

```python
N = 50
X = np.random.randn(N, 3, 32, 32)
y = np.random.randint(10, size=N)

loss, grads = model.loss(X, y)
print('Initial loss (no regularization): ', loss)

model.reg = 0.5
loss, grads = model.loss(X, y)
print('Initial loss (with regularization): ', loss)
```

### 方案二， 数值方法计算一阶梯度

数值方法的梯度即定义法， 当输入增加一个极小值时， 输出增加值比输入增加值的比率。
数值方法的问题是， 效率极低。 但可以作为分析法梯度的一个检验。已确定所用的分析梯度的正确性。

```python

from cs231n.gradient_check import eval_numerical_gradient_array, eval_numerical_gradient

num_inputs = 2
input_dim = (3, 16, 16)
reg = 0.0
num_classes = 10
np.random.seed(231)
X = np.random.randn(num_inputs, *input_dim)
y = np.random.randint(num_classes, size=num_inputs)

model = ThreeLayerConvNet(num_filters=3, filter_size=3,
                          input_dim=input_dim, hidden_dim=7,
                          dtype=np.float64)
loss, grads = model.loss(X, y)
for param_name in sorted(grads):
    f = lambda _: model.loss(X, y)[0]
    param_grad_num = eval_numerical_gradient(f, model.params[param_name], verbose=False, h=1e-6)
    e = rel_error(param_grad_num, grads[param_name])
    print('%s max relative error: %e' % (param_name, rel_error(param_grad_num, grads[param_name])))
```

其中， 数值梯度API的实现在这： 。 一般的， 二者的差异应小于0.001, 可能非常小。

### 方案三, Overfit小数据

对于一个小的数据集， 一个正确编写的深度模型， 往往可以很快的实现overfit, 即接近100%的training set的正确率。 但由于数据集很小， validation set的正确率往往很低， 形成非常明显的Overfitting, 这说明了你的模型编写是正确的。如下图:

![OverfittingSmallData](/img/OverfitSmallData.png)

可见， trainning accuracy 达到了100%, 但validation accuracy低于20%, 说明模型编写正确。

```python
np.random.seed(231)

num_train = 100
small_data = {
  'X_train': data['X_train'][:num_train],
  'y_train': data['y_train'][:num_train],
  'X_val': data['X_val'],
  'y_val': data['y_val'],
}

model = ThreeLayerConvNet(weight_scale=1e-2, dropout=0.75, use_batchnorm=True)

solver = Solver(model, small_data,
                num_epochs=15, batch_size=50,
                update_rule='adam',
                optim_config={
                  'learning_rate': 1e-3,
                },
                verbose=True, print_every=1)
solver.train()

plt.subplot(2, 1, 1)
plt.plot(solver.loss_history, 'o')
plt.xlabel('iteration')
plt.ylabel('loss')

plt.subplot(2, 1, 2)
plt.plot(solver.train_acc_history, '-o')
plt.plot(solver.val_acc_history, '-o')
plt.legend(['train', 'val'], loc='upper left')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.show()
```

### 方案四， 应用一些默认超参进行初识训练

选择一些常用的参数值， 对完整的数据和模型进行一次训练， 可以得到对于数据和模型的一个初识的大致认识。

以下为CIFAR-10的5000张图在三层卷积网络下的训练结果（默认参数, dropout possiblity = 0.55），正常模型， 带BN模型， 带BN+DropOut模型。

![res1](/img/tresWoReg.png)
![res2](/img/resWithBN.png)
![res3](/img/resWithBNDO.png)

可见：
1， BN的加入， 大大提高了训练集估计的准确率， 略微提高了验证集的准确率。

2， DropOut的加入， 降低了Overfitting, 提升了验证集的准确率。在默认参数未经调优的情况下， 已经可以达到近60%的val accuracy.

在此基础上， 可以考虑对reg, lr等参数进行调试， 已获得更好的结果。 后期我会专门再写一篇关于如何调参的。 这里不再赘述。

## 结语

通过这个系列的五篇文章， 我们从零开始用Python/Numpy实现了一个卷积网络。 在实际的工程中， 可能更多的还是使用Tensorflow等框架， 但自己实现一遍之后， 对整个模型的算法原理， 实现要点及工程技巧都有了更深刻的认识。

接下来， 我还会在此基础上， 实现更多的DL算法。

---

## 快速联系作者

欢迎关注我的知乎: <https://www.zhihu.com/people/NickWey> 


或直接在Github上联系我: <https://github.com/nick6918>

