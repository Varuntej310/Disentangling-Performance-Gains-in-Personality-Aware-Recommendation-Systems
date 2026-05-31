"""
Ablation studies and analysis
"""
import torch
import torch.nn.functional as F
import numpy as np
from collections import defaultdict

from training.evaluator import evaluate_embeddings
from models.graphsage_models import Model_concat, Model_graphsage


def get_embeddings_with_user_x(model, model_class, data_device, user_x_override=None):
    model.eval()
    with torch.no_grad():
        if user_x_override is None:
            user_x = data_device["user"].x
        else:
            user_x = user_x_override
        
        if model_class == Model_concat:
            user_embedding = model.user_emb(data_device["user"].node_id)
            user_features = F.relu(model.user_feat_lin(user_x))
            combined = torch.cat([user_embedding, user_features], dim=1)
            user_init = model.user_combiner(combined)
            x_dict = {
                "user": user_init,
                "movie": model.movie_emb(data_device["movie"].node_id),
            }
        elif model_class == Model_graphsage:
            # graphsage model without any auxiliary features
            x_dict = {
                "user": model.user_emb(data_device["user"].node_id),
                "movie": model.movie_emb(data_device["movie"].node_id),
            }
        else:
            # Linear and MTL both use additive form
            x_dict = {
                "user": model.user_emb(data_device["user"].node_id) + model.user_lin(user_x),
                "movie": model.movie_emb(data_device["movie"].node_id),
            }
        
        final_x = model.gnn(x_dict, data_device.edge_index_dict)
        
        return (
            final_x["user"].cpu().numpy(),
            final_x["movie"].cpu().numpy()
        )


def evaluate_personality_ablation(model, model_class, data_device, test_edges, 
                                   train_edges, config):
    train_set = defaultdict(set)
    for u, i in train_edges:
        train_set[int(u)].add(int(i))
    user_emb, movie_emb = get_embeddings_with_user_x(
        model, model_class, data_device, user_x_override=None
    )
    
    hr_real, ndcg_real = evaluate_embeddings(
        user_emb, movie_emb,
        test_edges, train_set,
        num_negatives=config.eval.num_negatives,
        K_list=config.eval.k_list,
        rng=np.random.RandomState(config.training.seed)
    )
    
    # Evaluate with zeroed personalities
    zero_x = torch.zeros_like(data_device["user"].x)
    
    user_emb_zero, movie_emb_zero = get_embeddings_with_user_x(
        model, model_class, data_device, user_x_override=zero_x
    )
    
    hr_zero, ndcg_zero = evaluate_embeddings(
        user_emb_zero, movie_emb_zero,
        test_edges, train_set,
        num_negatives=config.eval.num_negatives,
        K_list=config.eval.k_list,
        rng=np.random.RandomState(config.training.seed)
    )
    
    return {
        "real": {**hr_real, **ndcg_real},
        "zero": {**hr_zero, **ndcg_zero},
        "delta": {
            k: ndcg_real[k] - ndcg_zero[k]
            for k in ndcg_real
        }
    }


