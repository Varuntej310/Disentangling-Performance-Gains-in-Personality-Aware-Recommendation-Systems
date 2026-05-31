"""
LightGCN and NCF-style baseline models
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict


class LightGCN(torch.nn.Module):
    def __init__(self, num_users, num_movies, embedding_dim, num_layers=3):
        super().__init__()
        self.num_users = num_users
        self.num_movies = num_movies
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        
        self.user_emb = torch.nn.Embedding(num_users, embedding_dim)
        self.movie_emb = torch.nn.Embedding(num_movies, embedding_dim)
        
        torch.nn.init.normal_(self.user_emb.weight, std=0.1)
        torch.nn.init.normal_(self.movie_emb.weight, std=0.1)
    
    def forward(self, edge_index):
        U, M = self.num_users, self.num_movies
        emb = torch.cat([self.user_emb.weight, self.movie_emb.weight], dim=0)
        all_embs = [emb]
        
        row, col = edge_index
        col = col + U  # shift movies
        
        # Undirected
        indices = torch.cat([
            torch.stack([row, col], dim=0),
            torch.stack([col, row], dim=0)
        ], dim=1)
        
        deg = torch.bincount(indices[0], minlength=U + M).float()
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0.0
        
        norm = deg_inv_sqrt[indices[0]] * deg_inv_sqrt[indices[1]]
        
        A = torch.sparse_coo_tensor(
            indices, norm, (U + M, U + M)
        ).to(emb.device)

        for _ in range(self.num_layers):
            emb = torch.sparse.mm(A, emb)
            all_embs.append(emb)
        
        # Layer averaging
        final_emb = torch.mean(torch.stack(all_embs, dim=0), dim=0)
        
        return final_emb[:U], final_emb[U:]


def bpr_loss(user_emb, pos_emb, neg_emb):
    """Bayesian Personalized Ranking loss"""
    pos_score = (user_emb * pos_emb).sum(dim=1)
    neg_score = (user_emb * neg_emb).sum(dim=1)
    return -torch.mean(torch.log(torch.sigmoid(pos_score - neg_score) + 1e-8))