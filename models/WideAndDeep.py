import torch
import torch.nn as nn
from layers import Feature_Embedding, MLP, Feature_Embedding_Sum


class WideAndDeep(nn.Module):
    def __init__(self, feature_dims, embed_size, hidden_nbs, dropout=0):
        super(WideAndDeep, self).__init__()
        self.embedding = Feature_Embedding(feature_dims=feature_dims, embed_size=embed_size)
        self.embed_out_dim = len(feature_dims) * embed_size
        self.mlp = MLP(input_dim=self.embed_out_dim,
                       hidden_nbs=hidden_nbs,
                       out_dim=hidden_nbs[-1], last_act=None, drop_rate=dropout)
        self.linear = Feature_Embedding_Sum(feature_dims=feature_dims, out_dim=1)

    def forward(self, data):
        out_linear = self.linear(data)
        data = self.embedding(data).view(-1, self.embed_out_dim)
        out_mlp = self.mlp(data)
        out = torch.sigmoid(out_linear+out_mlp).squeeze(1)
        return out


data = torch.tensor([
    [1,2],
    [2,3],
    [1,3]
], dtype=torch.long)

model = WideAndDeep([10, 23], 12, [3, 4, 1], 0)
out = model(data)
print(out)