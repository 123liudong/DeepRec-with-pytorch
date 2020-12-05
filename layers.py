import torch
import numpy as np
import torch.nn as nn


class Feature_Embedding(nn.Module):
    '''
    实现特征的嵌入
    '''
    def __init__(self, feature_dims, embed_size):
        '''
        :param feature_dims: 各个特征的数量，如[3, 32, 343]表示特征1有3个取值范围，特征2有32个取值范围
        :param embed_size: 嵌入向量的维度，这里嵌入到同一个维度
        '''
        super(Feature_Embedding, self).__init__()
        self.embedding = nn.Embedding(sum(feature_dims), embedding_dim=embed_size)
        self.offset = np.array([0, *np.cumsum(feature_dims)[:-1]], dtype=np.long)
        torch.nn.init.xavier_uniform_(self.embedding.weight.data)

    def forward(self, data):
        '''
        :param data: Long Tensor
        :return:
        '''
        # unsqueeze主要是考虑到batch的存在
        data = data + torch.tensor(self.offset, dtype=torch.long).unsqueeze(0)
        return self.embedding(data)
# model = Feature_Embedding([3, 5], 16)
#
# data = torch.tensor([
#     [1, 2],
#     [0, 1],
#     [1, 1],
#     [0, 3]
# ], dtype=torch.long)
# out = model(data)


class Feature_Embedding_Sum(nn.Module):
    '''
    对特征向量化后，然后对所有向量求和，得到一个包含了所有信息的向量
    '''
    def __init__(self, feature_dims, out_dim=1):
        super(Feature_Embedding_Sum, self).__init__()
        self.embedding = nn.Embedding(sum(feature_dims), out_dim)
        self.bias = nn.Parameter(torch.zeros((out_dim,)))
        self.offset = np.array([0, *np.cumsum(feature_dims)[:-1]], dtype=np.long)

    def forward(self, data):
        '''
        :param data: Long Tensor
        :return:
        '''
        # unsqueeze主要是考虑到batch的存在
        data = data + torch.tensor(self.offset, dtype=torch.long).unsqueeze(0)
        print(self.embedding(data).size())
        # 把所有embedding之后的值向量叠加起来，得到一个向量
        data = torch.sum(self.embedding(data), dim=1) + self.bias
        return data
# model = Feature_Embedding_Sum([3, 5], 3)
# data = torch.tensor([
#     [1, 2],
#     [0, 1],
#     [1, 1],
#     [0, 3]
# ], dtype=torch.long)
# out = model(data)
# print(out.size())

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_nbs, out_dim, last_act='sigmoid', drop_rate=0.2):
        '''
        :param input_dim: 输入层的神经元个数
        :param hidden_nbs: 列表，存储的是各个隐藏层神经元的个数
        :param out_dim: 输出层的维度
        :param last_act: 输出层的激活函数 'sigmoid', 'softmax'
        :param drop_rate:
        '''
        super(MLP, self).__init__()
        layers = []
        for nb in hidden_nbs:
            layers.append(nn.Linear(input_dim, nb))
            layers.append(nn.BatchNorm1d(nb))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=drop_rate))
            input_dim = nb
        layers.append(nn.Linear(input_dim, out_dim))
        self.mlp = nn.Sequential(*layers)
        if last_act == 'sigmoid':
            self.mlp.add_module('sigmoid', nn.Sigmoid())
        elif last_act == 'softmax':
            self.mlp.add_module('softmax', nn.Softmax())

    def forward(self, data):
        return self.mlp(data)


model = MLP(3, [3,4,5], 1, last_act='sigmoid')
data = torch.randn((1024, 3))
print(model(data)[:10])