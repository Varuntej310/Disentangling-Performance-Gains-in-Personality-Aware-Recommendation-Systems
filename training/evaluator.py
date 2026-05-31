"""
Evaluation metrics for recommendation
"""
import numpy as np
from math import log2
from collections import defaultdict


def evaluate_embeddings(user_emb_np, movie_emb_np, test_edges, train_edges_set,
                        num_negatives=99, K_list=(3, 5, 10), rng=None):
    if rng is None:
        rng = np.random.RandomState(0)
    
    hits = {k: 0 for k in K_list}
    ndcg = {k: 0.0 for k in K_list}
    count = 0
    num_items = movie_emb_np.shape[0]
    
    for (u, pos) in test_edges:
        u = int(u)
        pos = int(pos)
        
        # Sample negative items
        negs = []
        while len(negs) < num_negatives:
            cand = int(rng.randint(0, num_items))
            if cand == pos or cand in train_edges_set[u]:
                continue
            negs.append(cand)
        
        # Score candidates
        candidates = [pos] + negs
        scores = movie_emb_np[candidates] @ user_emb_np[u]
        order = np.argsort(-scores)
        
        rank_of_pos = int(np.where(order == 0)[0][0])
        
        count += 1
        for K in K_list:
            if rank_of_pos < K:
                hits[K] += 1
                ndcg[K] += 1.0 / log2(rank_of_pos + 2)
    
    return (
        {f"HR@{K}": hits[K] / count for K in K_list},
        {f"NDCG@{K}": ndcg[K] / count for K in K_list}
    )