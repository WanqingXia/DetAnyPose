from torch import nn
import torch
import torch.nn.functional as F


class Similarity(nn.Module):
    def __init__(self):
        super(Similarity, self).__init__()

    def forward(self, query, reference):
        query = F.normalize(query, dim=-1)
        reference = F.normalize(reference, dim=-1)
        similarity = F.cosine_similarity(query, reference, dim=-1)
        return similarity.clamp(min=0.0, max=1.0)
