---
layout:     post
title:深度探索Deep_NLP及其Python实现(一):word2Vec的原理及其实现
date:       2018-04-22 22:00:00
author:     "nickiwei"
header-img: "img/post-bg-2015.jpg"
tags:
    - 深度学习
    - word2vec
    - NLP
---

这个系列从最基础的word2vec开始， 从零开始实现深度网络在NLP领域的各类模型及其应用。 本系列要求读者对深度神经网络有基础的理解（如全连接网络， 卷积网络）等。

欢迎转载， 转载请注明出处及链接。

完整代码库请查看我的GithubRepo: https://github.com/nick6918/MyNLP .部分代码参考了Stanford CS224n 课程作业。

## Before Word2Vec: WordNet

WordNet是一个词典语意库， 每一个语义被封装成了一个sense对象。 同一个word可能对应多个sense， 同一个sense也可以对应多个word. WordNet是传统NLP sentence understanding最经典的处理工具。每一个（word， sense）被称为一个lemma. 重点就是， wordNet是一个记录词与词关系的词库。

### Install WordNet

WordNet包含在Python自然语言处理的核心库nltk中， 首先确保你的环境中已经安装了nltk library.

	pip install nltk
	
此外，要想使用wordnet， 还需额外手动下载wordnet数据库。

	import nltk
	nltk.download('wordnet')
	
完成后进行测试

	from nltk.corpus import wordnet as wn
	wm.synsets("publish")
	
当你获得如下response即表明wordnet安装完成。

>>> wn.synsets('publish')
[Synset('print.v.01'), Synset('publish.v.02'), Synset('publish.v.03')]

### 利用WordNet计算语义距离

关于wordnet的详细文档， 可以直接查看[官方文档](http://www.nltk.org/howto/wordnet.html)， 这里介绍一种经典操作，即计算语义距离。

在wordnet中， 名词和动词组成了完整的语义层次， 可以计算(语义之间的)距离。

	sense1 = wn.synsets('recommend')[-1]
	sense2 = wn.synnets('suggest')[-1]
	sense1.shortest_path_distance(sense2)
	=> 0

而形容词和副词则难以获得定量的距离， 只能得到大致的相似词。

	a_sense = wn.synsets('glorious')[0]
	a_sense.similar_tos()
	=> [Synsets（‘devine.s.06’）, ...]
	
### WordNet的问题

作为一个静态词库， WordNet的核心问题，在于词的类型和数量有限且难于添加新词， 由于wordNet背后仍然是以one_not为表示形式的数据库， 因此，很容易就遇到了维度爆炸的问题。
为此， 我们引入了分布式词表示， Districubuted Word Representation. 而Word2Vec就是其中的典型代表。

# Word2Vec理解

## 模型基础与目标function

word2vec词向量的核心idea是， 我们希望把一个词作为已出现的词， 他周边的词出现在其周边的概率，如下图：
	
![w2v](/img/222.jpg)

由此， 我们可以容易的得到如下基本的loss function:

	J = 1 − p(w−t |wt) 
	
同时， 在构造W2C模型时， 我们分别initialize两个V*D的矩阵（V是词库数量， D是词向量维度）， 分别代表每一个词作为中心词和窗口词时的向量， 之所以划分成两个矩阵， 主要还是考虑到帮助模型收敛， 防止模型过度震荡。

训练完成后， 我们最终使用的词向量模型为两个矩阵之和（这是最简单有效的做法）

	Y = O + C

此时， 我们的目标函数objective function为

![objective_func](/img/W2C01.gif)

## 单样本损失函数

在对应的loss计算中， 我们有两种形式的loss可以应用：

简单的就是CE loss:

![objective_func](/img/W2C02.gif)

关于softmax cross entropy loss的深入分析， 可以查看这一系列的下一篇文章， 《深入分析softmax cross entropy loss》

更适合的是采用Negative Sampling loss， 其形式为：

![objective_func](/img/W2C03.jpg)

根据Markov et al在其论文中的叙述， 相比交叉商损失函数， 负采样损失函数的好处在于:

* 模型不仅仅是找到正确的分类， 此外， 还增强了模型对抗噪音样本的能力
* 在计算中， 负采样loss是更computing efficient的

由此， 我们可以分别计算出两种不同的损失函数对uo, uc, vc的梯度：

* CE loss

![objective_func](/img/W2C04.jpg)
![objective_func](/img/W2C05.jpg)

* Negtive Sampling loss

![objective_func](/img/W2C06.jpg)

具体计算如下（CE求梯， Neg同理不再详述）:

![objective_func](/img/W2C07.jpeg)

## 多样本损失函数与模型训练

注意， 以上的梯度计算均针对一组样本， 即一个中心词和一个对应的窗口词， 在实际模型训练时，我们采用的ngram模型要完成2n次这样的计算 。为此， 我们有两种训练方案：

### skip gram

Key idea:

将中心词作为先验， 用每一个ngram窗口词作为label， 然后将所有的窗口词与当前中心词的loss相加， 此时， 总的loss为：

![objective_func](/img/W2C09.jpg)
	
其中， O表示ngram窗口词， C表示中心词。 注意， 当我们把所有loss相加时， 在反向传播中， loss的梯度被完整的传递给了每一个窗口词， 同时， 中心词得到了n次等量的梯度更新。

### CBOW

Key idea：

CBOW的思路不是那么直观， 但是是一种计算上更有效率的做法， 我们将所有窗口词梯度相加， 然后与窗口词作一次softmax计算， 我们此时实际上在做的是， 利用ngram的所有窗口词去预估中心词， 此时的loss为:

![w2v](/img/W2C10.jpg)

此时， 反向传播中， 梯度首先传递给中心词， 和总和向量， 然后总和向量在分别传递给每一个窗口词， 因此， 中心词和每一个窗口词均更新一次梯度。

# Cost及梯度计算的实现

在实现了两种学习模型之后， 我们还需要为其设计合适的loss计算公式。 如上所述， 最常见的loss计算为Softmax Cross Entropy 和 Negative Sampling， 后者几乎可以看作是前者的性能优化。

### Softmax CrossEntropy

这里， 我们实现了之前设计的softmax objective function, loss function和三个梯度计算公式。最终返回 cost 和 input matrix, output matrix的梯度。

```python
def softmaxCostAndGradient(predicted, target, outputVectors, dataset=None):

    N, D = outputVectors.shape

    inner_products = np.dot(outputVectors, predicted)
    scores = softmax(inner_products) #(N, 1)
    cost = -np.dot(outputVectors[target], predicted) + np.log(np.sum(np.exp(inner_products)))

    gradPred = - outputVectors[target] + np.sum(scores.reshape(-1, 1) * outputVectors, axis=0)
    grad = scores.reshape(-1, 1)* np.tile(predicted, (N, 1))
    grad[target] -= predicted

    return cost, gradPred, grad
```
### Negative Sampling

在实现NegativeSampling的cost 和梯度计算之前， 我们首先实现一个负采样的辅助函数。在这个函数中， 我们从dataset中随机选取k个与target不一致的样本， 并输出index array.

```python
def getNegativeSamples(target, dataset, K):
    """ Samples K indexes which are not the target """

    indices = [None] * K
    for k in xrange(K):
        newidx = dataset.sampleTokenIdx()
        while newidx == target:
            newidx = dataset.sampleTokenIdx()
        indices[k] = newidx
    return indices
```

接下来， 我们利用之前设计的Negtive Sampling 的objective function， loss function和三个梯度公式， 最终返回cost 和 input matrix, output matrix的梯度。

```python
def negSamplingCostAndGradient(predicted, target, outputVectors, dataset,
                               K=10):
    indices = [target]
    indices.extend(getNegativeSamples(target, dataset, K))

    u0 = outputVectors[target]
    selected_vectors = outputVectors[indices]
    cost = -np.log(sigmoid(np.dot(u0, predicted)))-np.sum(np.log(sigmoid(np.dot(-selected_vectors, predicted))))

    #gradPred = selected_units_minus_one[0]*outputVectors[target] - np.sum(selected_units_minus_one[1:].reshape(-1, 1)*outputVectors[indices][1:], axis = 0)
    gradPred = (sigmoid(np.dot(u0, predicted)) - 1) * u0 - np.sum((sigmoid(np.dot(-selected_vectors, predicted))-1).reshape(-1, 1)*selected_vectors, axis=0)

    grad = np.zeros_like(outputVectors)
    grad_temp = -(sigmoid(np.dot(-selected_vectors, predicted))-1).reshape(-1, 1)*predicted
    grad[target] = (sigmoid(np.dot(u0, predicted)) - 1).reshape(-1, 1)*predicted
    np.add.at(grad, indices, grad_temp)

    assert gradPred.shape == predicted.shape
    assert grad.shape == outputVectors.shape

    return cost, gradPred, grad
```

注意我们这里是用了一个比较特殊的numpy函数np.add.at, 非常方便。

	np.add.at(grad, indices, grad_temp)

# Word2Vec的实现

我们已经实现了基于训练中的一组样本（一个窗口词和一个中心词）完成cost和梯度计算。接下来我们要针对一组ngram样本（即一个中心词和一组ngram窗口词）完成计算。

我们将分别实现skip gram和cbow, 值得注意的是， 我们引入word2vecCostAndGradient的函数参数， 使得两种模型在计算一组样本的cost和参数时， 可以自由选择softmax ce和negtive sampling两种算法。

首先实现skip-gram:

```python
def skipgram(currentWord, C, contextWords, tokens, inputVectors, outputVectors, dataset, word2vecCostAndGradient=softmaxCostAndGradient):
    """ Skip-gram model in word2vec
    
    currrentWord -- a string of the current center word
    C -- integer, context size
    contextWords -- list of no more than 2*C strings, the context words
    tokens -- a dictionary that maps words to their indices in
              the word vector list
    """

    cost = 0.0
    gradIn = np.zeros(inputVectors.shape)
    gradOut = np.zeros(outputVectors.shape)
    N, D = inputVectors.shape

    center_index = tokens[currentWord]
    predicted = inputVectors[center_index]
    for word in contextWords:
        target = tokens[word]
        cur_cost, cur_gradIn, cur_gradOut = word2vecCostAndGradient(predicted, target, outputVectors, dataset)
        cost += cur_cost
        gradIn[center_index] += cur_gradIn
        gradOut += cur_gradOut

    return cost, gradIn, gradOut
```

整个代码比较容易理解， 每一轮训练中， 都会有一个中心词， 和一组窗口词，对于每一个窗口词， 都会和中心词利用word2vecCostAndGradient做一次计算， 得到cost和梯度用于更新。

```
def cbow(currentWord, C, contextWords, tokens, inputVectors, outputVectors,
         dataset, word2vecCostAndGradient=softmaxCostAndGradient):
    cost = 0.0
    gradIn = np.zeros(inputVectors.shape)
    gradOut = np.zeros(outputVectors.shape)
    N, D = inputVectors.shape

    target = tokens[currentWord]
    contextWords_vectors = np.array([inputVectors[tokens[word]] for word in contextWords])
    predicted = np.sum(contextWords_vectors, axis = 0)
    cost, gradpred, grad = word2vecCostAndGradient(predicted, target, outputVectors, dataset)

    for word in contextWords:
        gradIn[tokens[word]] += gradpred
    gradOut = grad

    return cost, gradIn, gradOut
```

对比一下， CBOW将所有窗口词的inputVectors相加， 与中心词的outputVectors做卷积，输出梯度， 每个窗口词的inputVector梯度都被更新（包括中心词）， 而只有中心词的outVectors被更新。因为只做一次卷积， 所以cost也只有一个。 对比而言， Skip-gram要完成n个卷积， 每次卷积都是一个窗口词和中心词卷积， 中心词累积n个input梯度，每次迭代都有一个窗口词被更新梯度， cost也累加n次。总结如下图:

![grad](/img/grad.jpg)

可以看出， 相比Skip-Gram， CBOW只完成一次卷积， 却同时更新了三组数据， 效率更高些。在实际应用中， CBOW也确实更常见些， 但二者的性能差异并不显著。特别注意， 窗口词的OutputVector是<b>n个一起</b>增加一次。

# GloVe: 另一种思路

Word2Vec本质上是在估计词共现的概率， 提到概率， 我们自然就想到了频率。 我们可以直接利用词共现的频率来做词表达， 在解决维度爆炸问题上， 可以使用SVD做主成分分析， 进行降维。

![w2v](/img/glove.jpg)

但这种方法最大的问题是词矩阵训练算法受词个数影响， 难于scale. Glove将静态的词频统计和动态的词向量学习结合在一起， 解决了词频类算法的维度爆炸问题。Glove的具体细节， 可参考： [GloVe: Global Vectors for Word Representa:on (Pennington et al. (2014)](https://nlp.stanford.edu/pubs/glove.pdf)

Glove的loss function:

![w2v](/img/glove2.jpg)

GloVe和Word2Vec在大数据集上performance上相差无几， 但Glove更适合并行化操作。在小数据集上， Word2Vec表现出了更好的performanece.
