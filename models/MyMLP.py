import torch
import torch.nn as nn
from layers import Feature_Embedding, My_MLP


class MyMLP(nn.Module):
    def __init__(self, device, feature_dims, embed_size, hidden_nbs, drop_rate=0):
        super(MyMLP, self).__init__()
        self.features_nb = len(feature_dims)*embed_size
        self.embedding = Feature_Embedding(feature_dims, embed_size=embed_size, device=device)
        self.mlp = My_MLP(input_dim=self.features_nb, hidden_nbs=hidden_nbs, out_dim=1, last_act='sigmoid', drop_rate=drop_rate)

    def forward(self, data):
        data = self.embedding(data)
        data = data.view(-1, self.features_nb)
        out = self.mlp(data)
        return out.squeeze(1)

# data = torch.tensor([
#     [1,2],
#     [2,3],
#     [1,3]
# ], dtype=torch.long)
# device = torch.device('cpu')
# model = MyMLP(device, [10, 23], 12, [3, 4, 5], 0)
# out = model(data)
# print(out)