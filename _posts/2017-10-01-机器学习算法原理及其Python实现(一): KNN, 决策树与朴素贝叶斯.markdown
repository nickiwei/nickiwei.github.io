---
layout:     post
title: 机器学习算法原理及其Python实现(一): KNN,决策树与朴素贝叶斯
date:       2017-05-01 12:01:00
author:     "nickiwei"
header-img: "img/post-bg-2015.jpg"
tags:
    - 机器学习
    - Bayes
---


这个系列从最基础的K邻近算法开始， 逐一介绍了机器学习（不含神经网络）的各个典型算法， 以及一些常用的工程技术（如Adaboost, PCA, SVD等）， 并比较了各个算法的优劣和适用场景， 另外， 每个算法都配有一个mini project应用帮助理解。 这是本系列的第一篇， 主要介绍了K邻近， k决策树和朴素贝叶斯三种算法。

*欢迎转载， 转载请注明出处及链接。*

*完整代码库请查看我的GithubRepo: <https://github.com/nick6918/MyMachineLearning> .部分代码参考了stanford CS229以及《机器学习实践》示例代码。*

## 特别说明

1, 本系列中的code全部使用Python/Numpy编写， 在无特别说明的情况下， 未经验证的默认Python/Numpy提供的标准算法为时空复杂度最优的。

2, 为了与矩阵论中保持一致， 本文中的所有独立向量均为列向量， 特征矩阵中， 每一个数据为一个列向量。 所有依赖向量(或Label)均为横向量/List。

## k邻近KNN

### 基本特性
* 优点： 精确度高， 不需训练数据， 对异常值不敏感， 无数据输入假定
* 缺点： 时空复杂度均高
* 使用数据： 连续型， 离散型


### 算法概述
对每个数据

1, 计算数据与所有训练数据的距离

2，取前k个数据， 选择出现最多的返回其label

### 核心算法

* 该算法没有trariner

* 算法classifier

```python
def kNearestNeighbor_classifier(vector, data, labels, k=1):
	#require: data and vector has same amount of feature. 
	#require: all data should be normalized first
	distMatrix = (data - vector)**2
	dist = (distMatrix.sum(axis=0))**0.5
	sortedLabelIndices = dist.argsort()
	classCount = {}
	for index in range(k):
		votedLabel = labels[sortedLabelIndices[index], 0]
		classCount[votedLabel] = classCount.get(votedLabel, 0)+1
	maxLabel = None
	maxCount = 0
	for label in classCount.keys():
		if classCount[label] >= maxCount:
			maxLabel = label
			maxCount = classCount[label]
	return maxLabel
```

### 要点
数据需要正则化(Normalize)， 以消除不同数据绝对值差异对距离计算的影响。

```python
def autoNorm(data):
	minVals = data.min(axis=1)[..., newaxis]
	maxVals = data.max(axis=1)[..., newaxis]
	ranges = maxVals - minVals
	return (data*1.0 - minVals) / ranges
```

### 备注
1， k决策树是k邻近的直接优化。

2， 在此阶段， 我们对于测试数据的取得仍未按比例， 未使用cross_validation.
 
### 缺失值处理
-> 默认值法:
均值或-1

## K决策树

### 算法概述: ID3算法
1， 计算每个标签的香农商， 选择香农商的信息增益最大的

InfoGain(feature i) = baseEntrophy - Sigma(p(featureValue)*entropy(remainData))

2， 按此标签将数据进行划分

3， 递归， 递归的终点是数据集中的数据均属于同一标签 或 无feature可用

### 核心算法
* 算法trariner

```python
def createTree(data, featureNames):
	labels = data[-1]
	if isLabelAllSame(labels): 
		return labels[0] # return a label
	if len(data) <= 1:
		return majorityCount(data) # return a label
	currentFeatureIndex = chooseBestFeatureToSplit(data)
	
	currentFeatureName = featureNames[currentFeatureIndex]
	myTree = {currentFeatureName: {} }
	del(featureNames[currentFeatureIndex])

	allValue = set(data[currentFeatureIndex])
	for value in allValue:
		branchFeatureNames = featureNames[:]
		remainData = splitDataset(data, currentFeatureIndex, value)
		myTree[currentFeatureName][value] = createTree(remainData, branchFeatureNames)
	return myTree 
```
1, 树的每一个值是一个子树or一个label名.

* 算法classifier

```python
def kTrees_classifier(vector, myTree, featureNames):
	#requirement: tree should at least use one feature to devide.
	currentFeature = myTree.keys()[0]
	currentFeatureIndex = featureNames.index(currentFeature)
	secondDict = myTree[currentFeature]
	currentValue = vector[currentFeatureIndex, 0]
	for key in secondDict.keys():
		if key == currentValue:
			if type(secondDict[key])!=type(dict()):
				return secondDict[key]
			else:
				return kTrees_classifier(vector, secondDict[key], featureNames)
```
1, 在secondDict中判断， 是否为目标value， 如果是的话， 看返回值类型（叶结点 or 中间节点）

* 利用信息增益选择最佳feature

```python
def chooseBestFeatureToSplit(data):
	print data
	dataCount = len(data[0])
	featureCount = len(data)-1
	baseEntrophy = calculateEntropy(data)
	bestInfoGain = 0.0
	bestFeature = -1
	for featureIndex in range(featureCount):
		allValue = set(data[featureIndex, ...])
		newEntrophy = 0.0
		for featureValue in allValue:
			remainData = splitDataset(data, featureIndex, featureValue)
			remainEntrophy = calculateEntropy(remainData)
			prob = len(remainData[0])*1.0/dataCount
			newEntrophy += -prob*remainEntrophy
		infoGain = baseEntrophy - newEntrophy
		if infoGain > bestInfoGain:
			bestFeature = featureIndex
			bestInfoGain = infoGain
	print "Index", featureIndex
	return featureIndex
```

### 要点
1， 连续型数据需要离散化， 但即使离散化， 如果存在太多特征划分， 仍容易产生过拟合等其他问题。

2, 当无feature可用时， 利用最大频率法求得本组使用的label.

```python
def majorityCount(data):
	#require: data with more than one label but no feature
	totalData = len(data[0])
	labelCount = {}
	maxCount = 0
	maxLabel = None
	for label in data[0]:
		labelCount[label] = labelCount.get(label, 0) + 1
		if labelCount[label] > maxCount:
			maxCount = labelCount[label]
			maxLabel = label
		if maxCount > totalData*1.0/2:
			return maxLabel
	return maxLabel
```

### 缺失值处理
1， 选择特征时
平均值法： 在计算feature a的信息增益时， 若某数据feature a缺失， 则去掉数据求信息增益。 然后给最终结果乘以 n-1/n

2, 数据划分时
比例分配法， 按feature a划分， 若某一数据的feature a缺失， 则将该数据划入所有类别中， 但权重改为1/m

3, 分类时
1， cost function法
分别按各个子树分类后， 计算信息增益。 ？？？
2， 默认值法

## 朴素贝叶斯分类器

### 算法概述

词集模型：词向量的值 只有0/1这两种选择
词袋模型： 词向量的值 是词出现的数目。

词集模型和词袋模型 只区别于生成向量的方法。其他均一致。

核心就是Bayes条件概率， 详情可查看<机器学习中的概率论: Bayes学派>

1, 构造词频向量（分词袋和词集）， 利用labelList计算P(Ci)

2, 根据不同的label， 相加不同的wi, 除以 总wi 并取log， 从而计算logP(wi|Ci)

 p(wi|Ci) 是一个矩阵， 每个列向量对应一个label， 每一项均为所有该label的数据中wi出现的词频 ／ 该label的总词频 （记忆： 只考虑label为ci的数据， 与其他无关）

3， vector*logP(wi|ci)+log(p(Ci)) 得到 p(ci|wi)的后验概率， 由于我们只比较大小， 所以对于所有p(ci|wi)都一样的p(wi)可以省略计算。 选择最大后验概率的ci作为其label。

### 核心算法

* 算法trainer

```python
def trainNB0(trainMatrix, labelList):
	#value has only 1 and 0
	featureNumber = len(trainMatrix)
	dataNumber = len(trainMatrix[0])
	#First P(Ci) noted as pPrior
	labelCount = {}
	pPrior = {}
	for label in labelList:
		labelCount[label] = labelCount.get(label, 0) + 1
	for label in labelCount.keys():
		pPrior[label] = labelCount[label]*1.0/len(labelList)
	#Second P(wi|ci) noted as pLikelihood
	pNum = {}
	pDenom = {}
	logpLikelihood = {}
	for label in labelCount.keys():
		pNum[label] = ones((featureNumber,1))
		pDenom[label] = 2.0
	for i in range(dataNumber):
		currentData = trainMatrix[..., i:i+1]
		pNum[labelList[i]] += currentData
		pDenom[labelList[i]] += currentData[..., 0].sum()
	for label in labelCount.keys():
		logpLikelihood[label] = log(pNum[label]*1.0) - log(pDenom[label])
	return logpLikelihood, pPrior
```
1, 为了防止小数除法溢出， 我们使用log减法题代。

```python
logpLikelihood[label] = log(pNum[label]*1.0) - log(pDenom[label])
```
2, 由于采用了log算法， 所以不能出现pNum 和 PDenom为0的情况， 一般的 我们设置pNum的初值为1， pDenom的初值为2.

```python
for label in labelCount.keys():
    pNum[label] = ones((featureNumber,1))
    pDenom[label] = 2.0
```

3, PNum的样子 {label1: <列向量>, ...} 

=> 在label为label1的列向量中， wi的词频

   PDenom的样子 {label1: 标量, ...}
	
=> 在label为label1的列向量中的总词频

* classify算法

```python
def classifyNB(testMatrix, logpLikelihood, pPrior):
	#require: testMatrix should have same shape as logpLikelihood.
	numOfData = len(testMatrix[0])
	featureNumber = len(testMatrix)
	labelList = pPrior.keys()
	labelNumber = len(labelList)
	pPost = zeros((labelNumber, numOfData))
	bestLabel = []
	for i in range(labelNumber):
		label = labelList[i]
		pPost[i] = (testMatrix * logpLikelihood[label]).sum(axis=0)+log(pPrior[label])
	for i in range(numOfData):
		bestLabel.append(labelList[argmax(pPost[..., i])])
	return bestLabel
```

1， 用此事计算词条vector的后验概率  => vector*logP(wi|ci)+log(p(Ci))

### 要点

1， 利用词袋或词集模型计算词列表的词频向量，区别仅仅是词频为1/0 还是词列表中该词出现的总次数

```python
def words2Vec(data, vocabList):
	#word set model
	#@data: list of string
	dataCount = len(data)
	featureCount = len(vocabList)
	dataMatrix = zeros((featureCount, dataCount))
	for i in range(dataCount):
		for word in data[i]:
			if word not in vocabList: 
				print "word ", word, "not in vocab list"
			else:
				currentIndex = vocabList.index(word)
				dataMatrix[currentIndex, i] = 1
	return dataMatrix

def words2Bag(data, vocabList):
	#word bag model
	#@data: list of string
	dataCount = len(data)
	featureCount = len(vocabList)
	dataMatrix = zeros((featureCount, dataCount))
	for i in range(dataCount):
		for word in data[i]:
			if word not in vocabList: 
				print "word ", word, "not in vocab list"
			else:
				currentIndex = vocabList.index(word)
				dataMatrix[currentIndex, i] += 1 # =>唯一区别
	return dataMatrix
``` 

2, 测试集的选择 cross-validation

在决策树算法中， 我们之间选择一定比例的数据作为测试集， 这要求提供更多的数据。

在有限数据的情况下， 我们可以采用cross-validation的方法选择测试集。

Cross-Validation:

多次重复训练。 在每次训练中， 随机选择一定比例的数据作为测试集， 最终要各次迭代的测试结果平均后即为该算法的最终测试结果。

该方法优势：

1， 一定程度上抵消了 绝对选择法 训练数据对结果的影响。
2， 更大程度的利用了有效数据，达到近似performance所需的数据相对更少。

### 待改进的缺陷

1, stop wordList
大量无用词（高频词， 辅助词）如 hi, you, me , and等对结果有较大影响。 需单独处理。

2， feature独立性， 词与词之间往往不独立， 主要缺陷是需要更多数据。

---

## 快速联系作者

欢迎关注我的知乎: <https://www.zhihu.com/people/NickWey> 


或直接在Github上联系我: <https://github.com/nick6918>

