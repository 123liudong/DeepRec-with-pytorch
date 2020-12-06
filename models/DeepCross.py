import torch
import torch.nn as nn
from layers import Feature_Embedding, MLP, CrossNet


class DeepCross(nn.Module):
    def __init__(self, feature_dims, embed_size, hidden_nbs, num_layer=5, dropout=0):
        super(DeepCross, self).__init__()
        self.embedding = Feature_Embedding(feature_dims=feature_dims, embed_size=embed_size)
        self.embed_out_dim = len(feature_dims) * embed_size
        self.cross = CrossNet(input_dim=self.embed_out_dim, num_layers=num_layer)
        self.mlp = MLP(input_dim=self.embed_out_dim,
                       hidden_nbs=hidden_nbs,
                       out_dim=hidden_nbs[-1], last_act=None, drop_rate=dropout)
        self.fc = nn.Linear(hidden_nbs[-1]+self.embed_out_dim, 1)

    def forward(self, data):
        data = self.embedding(data).view(-1, self.embed_out_dim)
        out_mlp = self.mlp(data)
        out_cross = self.cross(data)
        data = torch.cat([out_cross, out_mlp], dim=-1)
        out = torch.sigmoid(self.fc(data)).squeeze(1)
        return out


# data = torch.tensor([
#     [1,2],
#     [2,3],
#     [1,3]
# ], dtype=torch.long)
#
# model = DeepCross([10, 23], 12, [3, 4, 5], 0, 1)
# out = model(data)
# print(out)