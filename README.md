# Algo-with-pytorch
## 简介

 这个项目使用pytorch实现了一些使用深度推荐算法，这个项目首先是为了方便自己不用重复撸代码，其次便是为大家提供算法在代码层面上
 的一些参考，当然建议你直接拿过去使用。

 因为个人水平能力有限，代码可能存在错误，如果你发现了，那么请告知我改正哦~

## 需要的环境

- python3.7+
- numpy
- pytorch
- pandas
- ...

## 已实现的算法

- ...

## 待实现的算法
- CFGAN
- CDAE
- NeuMF
- NFM
- CML
- ...

## 关于项目结构

项目主要分为 4 个部分:
1. **data**: 数据处理包
2. **metrics**：计算相关的指标（具体指标请下滑查看）
3. **deep_models**：存放各种模型的包，可以开箱即用哦~
4. **run**：项目运行的入口
5. **dataset**: 数据集存放的目录，这里的数据集是自己处理过后的数据，具体说明请往下看。

## 关于 data 部分
- 推荐算法经常用到的数据包括userId, itemId以及用户对项目的score,普遍来说都是把这三个属性放在一个文件里!
这里也一样，格式如下：
```
userId, itemId, score
0, 0, 2
1, 2, 2
2, 0, 5
...
```
- 本项目不包含划分测试集和训练集的工具，需要自己划分数据集

- 本项目下标和用户下标都是从0开始，在此特意说明

## 关于 metrics 部分
ps. 如果你无法预览公式，下载这个[插件](https://chrome.google.com/webstore/detail/mathjax-plugin-for-github/ioemnmodlmafdkllaclgeombjnmnbima/related)即可

- rmse
$$
rmse = \sqrt{\frac{\sum_{u, i\in T}(r_{u, i}-\hat{r}_{u, i})^2}{|T|}}
$$

其中，$T$表示$u$和$i$的组合
