import torch
import torch.nn.functional as F

def cosine_similarity(ground_set, candidate):
    ground_norm = F.normalize(ground_set, p=2, dim=1)
    candidate_norm = F.normalize(candidate, p=2, dim=1)
    return torch.matmul(ground_norm , candidate_norm.T)

def euclidean_similarity(ground_set, candidate, sigma=1.0):
    dist = torch.cdist(ground_set, candidate, p=2) ** 2
    sim = torch.exp(-dist / (2 * sigma ** 2))
    return sim

def get_similarity_fn(metric='cosine', sigma=1.0):
    if metric == 'cosine':
        return cosine_similarity
    elif metric == 'euclidean':
        return lambda ground_set, candidate: euclidean_similarity(ground_set, candidate, sigma)
    else:
        raise ValueError(f"Unknown similarity metric: {metric}")