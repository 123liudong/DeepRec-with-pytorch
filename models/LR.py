import torch
import numpy as np
import torch.nn as nn
from layers import Feature_Embedding_Sum


class LR(nn.Module):
    def __init__(self, feature_dims):
        super(LR, self).__init__()
        self.predict = nn.Sequential(
            Feature_Embedding_Sum(feature_dims, out_dim=1),
            nn.Sigmoid()
        )

    def forward(self, data):
        return self.predict(data).squeeze(1)