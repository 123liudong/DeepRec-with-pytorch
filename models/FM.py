import torch
import torch.nn as nn
from layers import Feature_Embedding, Feature_Embedding_Sum, FactorizationMachine


class FM(nn.Module):
    def __init__(self, device, feature_dims, embed_size):
        super(FM, self).__init__()
        self.embedding = Feature_Embedding(feature_dims=feature_dims, embed_size=embed_size, device=device)
        self.fc = Feature_Embedding_Sum(feature_dims=feature_dims, out_dim=1, device=device)
        self.fm = FactorizationMachine(reduce_sum=True)

    def forward(self, data):
        embedding_features = self.embedding(data)
        out = self.fc(data) + self.fm(embedding_features)
        return torch.sigmoid(out.squeeze(1))

# data = torch.tensor([
#     [1,2],
#     [2,3],
#     [1,3]
# ], dtype=torch.long)
#
#
# model = FM([10, 23], 12)
# out = model(data)
# print(out)