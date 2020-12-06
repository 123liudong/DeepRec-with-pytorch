import torch
import torch.nn as nn
from layers import Feature_Embedding, Feature_Embedding_Sum, FactorizationMachine, MLP


class NFM(nn.Module):
    def __init__(self, feature_dims, embed_size, hidden_nbs, dropout):
        super(NFM, self).__init__()
        self.embedding = Feature_Embedding(feature_dims=feature_dims, embed_size=embed_size)
        self.fc = Feature_Embedding_Sum(feature_dims=feature_dims, out_dim=1)
        self.fm = FactorizationMachine(reduce_sum=False)
        self.mlp = MLP(input_dim=embed_size,
                       hidden_nbs=hidden_nbs,
                       out_dim=1, last_act=None, drop_rate=dropout)


    def forward(self, data):
        embedding_features = self.embedding(data)
        cross_result = self.fm(embedding_features)
        out = self.fc(data) + self.mlp(cross_result)
        return torch.sigmoid(out.squeeze(1))