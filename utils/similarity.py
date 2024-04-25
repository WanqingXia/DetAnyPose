import torch
from torch import nn
import torch.nn.functional as F


class CosineSimilarity(nn.Module):
    def __init__(self):
        super(CosineSimilarity, self).__init__()

    def forward(self, query, reference):
        query = F.normalize(query, dim=-1)
        reference = F.normalize(reference, dim=-1)
        similarity = F.cosine_similarity(query, reference, dim=-1)
        return similarity.clamp(min=0.0, max=1.0)


class L2Distance(nn.Module):
    def __init__(self):
        super(L2Distance, self).__init__()

    def forward(self, query, reference):
        # Compute the L2 distance between the query and reference embeddings
        distance = torch.norm(query - reference, p=2, dim=-1)
        return distance


class L1Distance(nn.Module):
    def __init__(self):
        super(L1Distance, self).__init__()

    def forward(self, query, reference):
        # Compute the L1 distance between the query and reference embeddings
        distance = torch.abs(query - reference).sum(dim=-1)
        return distance


