# rs-with-pytorch
实现一些常见的推荐算法模型，避免做重复性的工作

## 数据集
- movielens 1m

## 模型
- LR
- NCF
- FM
- NFM
- Wide&Deep
- DeepCross

## todo list
...


## 评价指标
### 回归
- rmse
- mae
- auc
### 分类
...

## 项目结构

- layers.py
- examples
- models
    - LR
    - NCF
    - FM
    - NFM
    - Wide&Deep
    - DeepCross
- utils
    - dataset
        - movielens.py
    - evaluate
        - regression_metrics.py
        - class_metrics.py