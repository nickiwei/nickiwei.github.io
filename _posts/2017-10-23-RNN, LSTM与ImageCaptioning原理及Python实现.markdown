---
layout:     post
title: RNN, LSTM与ImageCaptioning原理及Python实现
date:       2017-09-16 12:01:00
author:     "nickiwei"
header-img: "img/post-bg-2015.jpg"
tags:
    - 深度学习
    - RNN
    - LSTM
---

与上一系列《CNN卷积网络的Python实现》一致， 这一篇中， 我们将用pure python/Numpy实现RNN, LSTM， 并基于Microsoft COCO dataset实现ImageCaptioning.

*欢迎转载， 转载请注明出处及链接。*

*完整代码库请查看我的GithubRepo: <https://github.com/nick6918/MyDeepLearning> .部分代码参考了Stanford CS231n 课程作业。*


## Word Embeding

这一章， 我们要开始和一些自然语言打交道。 首先要解决的就是如何represent一个句子(word sequence).

### One hot representation

一个N长向来(N为总词数)， 每个单词对应的向量都是一个1+N-1个0的向量。

问题：

1， 向量维度爆炸

2， 不能很好的体现单词之间的相似性(如girl 与 girlfriend)，同词性性(如a 与 an)等相关性

3， 词feature不易确定

### Word Embeding 的方法进行自适应的学习

为了减少维度爆炸， 同时在词汇的表示时， 使用一些feature来表示特征。 我们希望这些特征是系统自己学习到的（与CNN一致）。由此，我们构造了如下的数据：

#### caption: (N, T) 取值范围: [0, V)

容易理解， 对于一个长度为T的句子(caption), 每一项的取值都是一个词。

#### W: 嵌套矩阵 (V, W)

嵌套矩阵为每一个词引入一个供学习的W向量， 表示该词在句子中的特征数据供系统学习使用。

#### X: 嵌套词矩阵 (N, T, W)

嵌套词的输出， 在RNN中与型为(W, H)的矩阵卷积， 贡献给h.

综上， 易得代码：

```python
def word_embedding_forward(x, W):
    """
    Forward pass for word embeddings. We operate on minibatches of size N where
    each sequence has length T. We assume a vocabulary of V words, assigning each
    to a vector of dimension D.
    """
    out = W[x, :]
	cache = None
    return out, cache
```

求梯度

```python
def word_embedding_backward(dout, cache):
    """
    Backward pass for word embeddings. We cannot back-propagate into the words
    since they are integers, so we only return gradient for the word embedding
    matrix.
    """
    x, W = cache
    dW = np.zeros_like(W)
    np.add.at(dW, x, dout)

    return dW
```
这是我们遇到的一个新的深度学习操作（嵌套）， 它与之前的卷积，pooling, element-wise乘积， activation, softmax等均不同，要特别注意它的梯度， 思想容易理解， 但np.add.at的使用还是很tricky的。

一句话理解的话， 就是embedding word提供了一个维度远低于词数量的基于特征的词表示法， 且特征是自学习的。

## Hidden State

除了caption输入外， 我们还需要输入h给RNN， 表示前序的sequence对输出的影响。

h 是一个(N, H)的矩阵， 初始的h0由features卷积而来。

之后的h由前序h和当前caption共同决定， 输出下一个train caption. 

 ```python
 next_h = np.tanh(x.dot(Wx) + prev_h.dot(Wh) + b)
 ```
 
## RNN Layer的实现

有了embedded word和hidden state的概念， 我们就可以实现rnn layer了。
由于RNN是一个时序模型， 我们需要首先实现一个rnn_step

```python
def rnn_step_forward(x, prev_h, Wx, Wh, b):
    """
    Run the forward pass for a single timestep of a vanilla RNN that uses a tanh
    activation function.

    The input data has dimension D, the hidden state has dimension H, and we use
    a minibatch size of N.
    """
    next_h = np.tanh(x.dot(Wx) + prev_h.dot(Wh) + b)
    cache = Wx, x, Wh, prev_h, next_h
    return next_h, cache
    
def rnn_step_backward(dnext_h, cache):
    """
    Backward pass for a single timestep of a vanilla RNN.
    """
    Wx, x, Wh, prev_h, next_h = cache
    N, H = dnext_h.shape
    dLoss_dH = (1 - next_h**2) * dnext_h #(N, H)
    dx = dLoss_dH.dot(Wx.T)
    dWx = x.T.dot(dLoss_dH)
    dprev_h = dLoss_dH.dot(Wh.T)
    dWh = prev_h.T.dot(dLoss_dH)
    db = np.ones(N).dot(dLoss_dH)
    return dx, dprev_h, dWx, dWh, db
```

前向的实现都很简单， 主要就是围绕着hidden state做一步卷积。 Backward注意 tanh的backward pass:

	dLoss_dH = (1 - next_h**2) * dnext_h    #(N, H)
有了一步的RNN， 我们就可以实现整个时间序列的RNN, 我们规定， 每个时间点time step产生一个word.所以， 对于一个长度为T的序列， 需要T步。

### Forward Pass

```python
def rnn_forward(x, h0, Wx, Wh, b):
    """
    Run a vanilla RNN forward on an entire sequence of data. We assume an input
    sequence composed of T vectors, each of dimension D. The RNN uses a hidden
    size of H, and we work over a minibatch containing N sequences. After running
    the RNN forward, we return the hidden states for all timesteps.
    """
    N, T, D = x.shape
    N, H = h0.shape
    h = np.zeros((N, T, H))
    prev_h = h0
    for i in range(T):
        current_x = x[:, i, :] #(N, D)
        prev_h, _ = rnn_step_forward(current_x, prev_h, Wx, Wh, b)
        h[:, i, :] = prev_h
    cache = Wx, x, Wh, h, h0

    return h, cache
``` 

### Backward Pass

```python
def rnn_backward(dh, cache):
    """
    Compute the backward pass for a vanilla RNN over an entire sequence of data.
    """
    Wx, x, Wh, h, h0 = cache
    N, T, H = dh.shape
    dx = np.zeros_like(x)
    dWx = np.zeros_like(Wx)
    dWh = np.zeros_like(Wh)
    db = np.zeros(H)
    dprev_h = 0
    for i in range(T-1, -1, -1):
        next_h = h[:, i, :]
        if i > 0:
            prev_h = h[:, i-1, :]
        else:
            prev_h = h0
        dnext_h = dh[:, i, :] + dprev_h
        current_cache = Wx, x[:, i, :], Wh, prev_h, next_h
        dx_i, dprev_h, dWx_i, dWh_i, db_i = rnn_step_backward(dnext_h, current_cache)
        dWx += dWx_i
        dWh += dWh_i
        db += db_i
        dx[:, i, :] = dx_i
    dh0 = dprev_h
    return dx, dh0, dWx, dWh, db
```
注意事项：

	dnext_h = dh[:, i, :] + dprev_h

分支的梯度是分支梯度的和。 这一点有时候比较容易出问题， 由于在数学中不存在赋值操作， 在求梯度时， 我们<b>把赋值x理解成 0 + x</b>，这样就可以看出， 在前向中， 每次求出的next\_h, 都与h（的某一行）做了求和， 同时用来计算下一个next_h, 所以后向中， 就需要把这两部分加起来。

## RNN的实现

与CNN一致， 我们在实现RNN时， 对于affine layer, softmax等的输入的shape， 需要做一些修改。这里不再详述.

```python
def loss(self, features, captions):
        """
        Compute training-time loss for the RNN. We input image features and
        ground-truth captions for those images, and use an RNN (or LSTM) to compute
        loss and gradients on all parameters.

        Inputs:
        - features: Input image features, of shape (N, D)
        - captions: Ground-truth captions; an integer array of shape (N, T) where
          each element is in the range 0 <= y[i, t] < V

        Returns a tuple of:
        - loss: Scalar loss
        - grads: Dictionary of gradients parallel to self.params
        """
        # Cut captions into two pieces: captions_in has everything but the last word
        # and will be input to the RNN; captions_out has everything but the first
        # word and this is what we will expect the RNN to generate. These are offset
        # by one relative to each other because the RNN should produce word (t+1)
        # after receiving word t. The first element of captions_in will be the START
        # token, and the first element of captions_out will be the first word.
        captions_in = captions[:, :-1]
        captions_out = captions[:, 1:]

        mask = (captions_out != self._null)

        # Weight and bias for the affine transform from image features to initial
        # hidden state
        W_proj, b_proj = self.params['W_proj'], self.params['b_proj']

        # Word embedding matrix
        W_embed = self.params['W_embed']

        # Input-to-hidden, hidden-to-hidden, and biases for the RNN
        Wx, Wh, b = self.params['Wx'], self.params['Wh'], self.params['b']

        # Weight and bias for the hidden-to-vocab transformation.
        W_vocab, b_vocab = self.params['W_vocab'], self.params['b_vocab']

        reg = self.reg
        loss, grads = 0.0, {}
        
        #1, compute initial hidden state: (N, D)*(D, H) = (N, H)
        #N: sample number
        #D: CNN final dim
        #H: Hidden dim for h
        h0, cache1 = affine_forward(features, W_proj, b_proj)

        #2, word embedding: (N, T) :* (V, W) = (N, T, W)
        #T: caption size(sentence length)
        #V: vocab_size
        #W: word_vec size
        x, cache2 = word_embedding_forward(captions_in, W_embed)

        #3, RNN or LSTM
        if self.cell_type == "rnn":
            #for t: (N, H) * (H, H) = (N, H)
            #for t: (N, W) * (W, H) = (N, H)
            #(N, T, W) -> (N, T, H)
            h, cache3 = rnn_forward(x, h0, Wx, Wh, b)
        else:
            pass

        #4, calculate vocab score(UNEMBEDDING): (N, T, W) * (W, V) = (N, T, V)
        scores, cache4 = temporal_affine_forward(h, W_vocab, b_vocab)

        #5, calculate loss
        loss, dLoss_dScores = temporal_softmax_loss(scores, captions_out, mask)
        dLoss_dh, grads["W_vocab"], grads["b_vocab"] = temporal_affine_backward(dLoss_dScores, cache4)
        grads["W_vocab"] += reg*W_vocab
        if self.cell_type == "rnn":
            dLoss_dx, dLoss_dh0, grads["Wx"], grads["Wh"], grads["b"] = rnn_backward(dLoss_dh, cache3)
        else:
            pass
        grads["Wx"] += reg*Wx
        grads["Wh"] += reg*Wh
        grads["W_embed"] = word_embedding_backward(dLoss_dx, cache2)
        grads["W_embed"] += reg*W_embed
        dLoss_dfeas, grads["W_proj"], grads["b_proj"] = affine_backward(dLoss_dh0, cache1)
        grads["W_proj"] += reg*W_proj
        return loss, grads
```

与CNN一致， 在调用模型前， 需要首先对模型进行初始化， 主要是对各种参数初始化。 这里不再详述。 在loss中， 我们定义了模型的前后向逻辑， 前向为:

```
1, 利用features计算h0
2, 利用caption产生词嵌套模型
3, RNNorLSTM, 产生最终的结果(N, T, H)
4, 将Hidden State卷积成词汇表V的scores(N, T, V)
5, 计算loss， 求反向梯度
```

## LSTM的引入

### 梯度爆炸or 梯度消失

![h0_Cal](/Img/h0gre.png)

在求h0 or x1等的梯度时， 需要与W多次相乘， 容易引起梯度爆炸or消失。

对于梯度爆炸， 可以通过设置gredient threshhold解决， 但梯度消失则无法轻易解决。

### Manage Gredient Fast Pass,  Improve performance

通过设置gredient的fast pass， 提高反向传播的效率。

## LSTM layer的实现

![LSTM](/Img/lstm.png)

可见， 每个x和h梯度的求取， 都可以通过当前c的梯度得到， 且c的梯度只与下一个c有关。

下图详细介绍了LSTM的具体计算：

![LSTM_CAL](/Img/lstm_cal.png)

基于上述计算， 我们实现LSTM layer， 与RNN一致， 我们首先实现一个lstm_step layer

```python
def lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b):
    """
    Forward pass for a single timestep of an LSTM.

    The input data has dimension D, the hidden state has dimension H, and we use
    a minibatch size of N.
	"""
    N, H = prev_h.shape
    middle_state = prev_h.dot(Wh) + x.dot(Wx) + b       #(N, 4H)
    i0, f0, o0, g0 = middle_state.T.reshape(4, H, -1)       #(H, N)
    i = sigmoid(i0.T)
    f = sigmoid(f0.T)
    g = np.tanh(g0.T)
    o = sigmoid(o0.T)
    next_c = prev_c * f + i * g
    next_h = o * np.tanh(next_c)
    cache = f, g, i, o, prev_h, prev_c, next_c, x, Wh, Wx

    return next_h, next_c, cache
    
def lstm_step_backward(dnext_h, dnext_c, cache):
    """
    Backward pass for a single timestep of an LSTM.
    """
   
    f, g, i, o, prev_h, prev_c, next_c, x, Wh, Wx = cache
    N, H = dnext_h.shape

    do = dnext_h * np.tanh(next_c)
    dnext_c += dnext_h * o * (1 - np.tanh(next_c)**2)
    dprev_c = dnext_c * f
    di = dnext_c * g
    dg = dnext_c * i
    df = dnext_c * prev_c
    #Gredient of tanh and sigmoid
    dg0 = (1 - g**2) * dg 
    df0 = df * (f*(1-f))   
    di0 = di * (i*(1-i))
    do0 = do * (o*(1-o)) 
    dmiddle_state = np.hstack((di0, df0, do0, dg0))   #(N, 4H)
    dWx = x.T.dot(dmiddle_state)
    dWh = prev_h.T.dot(dmiddle_state) 
    dprev_h = dmiddle_state.dot(Wh.T)
    dx = dmiddle_state.dot(Wx.T)
    db = np.ones(N).dot(dmiddle_state)

    return dx, dprev_h, dprev_c, dWx, dWh, db
```

与RNN一致， 我们计算T时间序列下的LSTM layer:

```python
def lstm_forward(x, h0, Wx, Wh, b):
    """
    Forward pass for an LSTM over an entire sequence of data. We assume an input
    sequence composed of T vectors, each of dimension D. The LSTM uses a hidden
    size of H, and we work over a minibatch containing N sequences. After running
    the LSTM forward, we return the hidden states for all timesteps.

    Note that the initial cell state is passed as input, but the initial cell
    state is set to zero. Also note that the cell state is not returned; it is
    an internal variable to the LSTM and is not accessed from outside.
	"""
    N, T, D = x.shape
    N, H = h0.shape
    next_c = np.zeros((N, H))
    next_h = h0
    h = np.zeros((N, T, H))
    caches = []
    for index in range(T):
        current_x = x[:, index, :]
        next_h, next_c, cache = lstm_step_forward(current_x, next_h, next_c, Wx, Wh, b)    
        caches.append(cache[:-2])
        h[:, index, :] = next_h
    cache = (caches, Wx, Wh)

    return h, cache


def lstm_backward(dh, cache):
    """
    Backward pass for an LSTM over an entire sequence of data.]
    """

    caches, Wx, Wh = cache
    N, T, H = dh.shape
    D, _ = Wx.shape
    dWx = np.zeros(Wx.shape)
    dWh = np.zeros(Wh.shape)
    dx = np.zeros((N, T, D))
    db = np.zeros(4*H)
    dnext_c = np.zeros((N, H))
    dnext_h = np.zeros((N, H))
    for index in range(T-1, -1, -1):
        dnext_h += dh[:, index, :]
        current_cache = caches[index] + (Wh, Wx)
        dx_i, dnext_h, dnext_c, dWx_i, dWh_i, db_i = lstm_step_backward(dnext_h, dnext_c, current_cache)
        dx[:, index, :] += dx_i
        dWh += dWh_i
        dWx += dWx_i
        db += db_i
    dh0 = dnext_h

    return dx, dh0, dWx, dWh, db

```

## RNN/LSTM的训练

在调用RNN模型是， 我们设置cell type 参数， 在rnn和lstm之间进行选择。并用rnn\_solver来

```python
med_lstm_model = CaptioningRNN(
          cell_type='lstm', #rnn or lstm
          word_to_idx=med_data['word_to_idx'],
          input_dim=med_data['train_features'].shape[1],
          hidden_dim=512,
          wordvec_dim=256,
          dtype=np.float32,
        )
        
med_lstm_solver = CaptioningSolver(med_lstm_model, med_data,
           update_rule='adam',
           num_epochs=50,
           batch_size=50,
           optim_config={
             'learning_rate': 5e-3,
           },
           lr_decay=0.995,
           verbose=True, print_every=10,
         )
                 
med_lstm_solver.train()
```

关于训练solver的具体实现， 大致与CNN solver一致， 读者可自行查看其源代码, Link: <https://github.com/nick6918/MyDeepLearning/blob/master/lib/solvers/captioning_solver.py>

完成训练后， 即可以用来实现Image Captioning. 测试代码如下：

```python
for split in ['train', 'val']:
    minibatch = sample_coco_minibatch(med_data, split=split, batch_size=2)
    gt_captions, features, urls = minibatch
    gt_captions = decode_captions(gt_captions, data['idx_to_word'])

    sample_captions = med_lstm_model.sample(features)
    sample_captions = decode_captions(sample_captions, data['idx_to_word'])

    for gt_caption, sample_caption, url in zip(gt_captions, sample_captions, urls):
        plt.imshow(image_from_url(url))
        plt.title('%s\n%s\nGT:%s' % (split, sample_caption, gt_caption))
        plt.axis('off')
        plt.show()
```

在使用默认参数的前提下， 我们大概能够达到90% train accuracy， 但只有30% - 40%的val accuracy， 除了可以进一步tuning parameter之外， 更重要的是需要引入更多数据， 同时， 引入一些正则化机制来抑制OverFitting, 如RNN DropOut等技术， 在此不再详述。

## 结语

这是Deep Learning Python实现的 最后一篇， 通过总计六篇文章， 我们从零开始了解了深度学习的思想， 常用layer， 构造了DNN, CNN, RNN等基本模型， 并尝试对CIFAR-10进行classfication， 以及对Microsoft coco实现Image Captioning。

从下一篇开始， 我们将使用TensorFlow来实现一些更复杂的模型和功能， 并尝试解决诸如Object Localization, Detection等更普遍的图像工程问题。

---

## 快速联系作者

欢迎关注我的知乎: <https://www.zhihu.com/people/NickWey> 


或直接在Github上联系我: <https://github.com/nick6918>
