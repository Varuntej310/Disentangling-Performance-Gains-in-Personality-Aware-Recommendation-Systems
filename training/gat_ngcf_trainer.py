"""
Training functions for GAT and NGCF models
"""
import torch
import torch.nn.functional as F
import numpy as np
import copy
from collections import defaultdict
from torch_geometric.loader import LinkNeighborLoader

from .evaluator import evaluate_embeddings
from models.gat_models import GAT_concat, GAT_attention_guided
from models.ngcf_models import NGCF_concat, NGCF_propagation_aware


def train_gat_ngcf_model(
    model_class,
    data,
    train_edges,
    val_edges,
    test_edges,
    config,
    personality_loss_weight=0.0,
    verbose=True
):
    torch.manual_seed(config.training.seed)
    np.random.seed(config.training.seed)
    
    device = torch.device(config.training.device if torch.cuda.is_available() else 'cpu')
    data_device = data.to(device)
    edge_label_index = torch.tensor(train_edges.T, dtype=torch.long)
    edge_label = torch.ones(edge_label_index.size(1), dtype=torch.float)
    
    train_loader = LinkNeighborLoader(
        data=data_device,
        num_neighbors=config.model.num_neighbors,
        neg_sampling_ratio=config.training.neg_sampling_ratio,
        edge_label_index=(("user", "rates", "movie"), edge_label_index),
        edge_label=edge_label,
        batch_size=config.training.batch_size,
        shuffle=True,
    )

    user_train_set = defaultdict(set)
    for u, i in train_edges:
        user_train_set[int(u)].add(int(i))
    
    num_users = data_device["user"].num_nodes
    num_movies = data_device["movie"].num_nodes
    
    model = model_class(
        hidden_channels=config.model.hidden_channels,
        num_users=num_users,
        num_movies=num_movies
    ).to(device)

    model.set_metadata(data_device.metadata())
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config.training.lr)
    
    best_val_score = -np.inf
    best_state = None
    epochs_no_improve = 0
    
    # Use personality loss weight
    if personality_loss_weight == 0.0 and hasattr(config.model, 'personality_loss_weight'):
        personality_loss_weight = config.model.personality_loss_weight
    
    for epoch in range(1, config.training.num_epochs + 1):
        model.train()
        total_loss = 0.0
        total_examples = 0
        
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            out = model(batch)
            
            if isinstance(out, dict):  # MTL model
                pred_link = out["link"]
                pred_personality = out["personality"]
                
                gt_link = batch["user", "rates", "movie"].edge_label
                user_ids = batch["user"].n_id
                gt_personality = data_device["user"].x[user_ids]
                
                loss_link = F.binary_cross_entropy_with_logits(pred_link, gt_link)
                loss_pers = F.mse_loss(pred_personality, gt_personality)
                loss = loss_link + personality_loss_weight * loss_pers
            else:
                gt_link = batch["user", "rates", "movie"].edge_label
                loss = F.binary_cross_entropy_with_logits(out, gt_link)
            
            loss.backward()
            optimizer.step()
            
            total_loss += float(loss) * gt_link.numel()
            total_examples += gt_link.numel()
        
        avg_loss = total_loss / total_examples
        
        if verbose:
            print(f"Epoch {epoch:03d} | Train Loss {avg_loss:.4f}", end="")
        
        # Validation
        val_score = evaluate_gat_ngcf_full_graph(
            model, model_class, data_device, val_edges, user_train_set,
            config, epoch, verbose
        )
        
        # Early stopping
        if val_score > best_val_score + 1e-6:
            best_val_score = val_score
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
            if verbose:
                print(f" | Val NDCG@10 {val_score:.4f} ✓")
        else:
            epochs_no_improve += 1
            if verbose:
                print(f" | Val NDCG@10 {val_score:.4f} (no improve: {epochs_no_improve})")
            if epochs_no_improve >= config.training.patience:
                if verbose:
                    print("Early stopping triggered.")
                break
    
    # Restore best
    if best_state is not None:
        model.load_state_dict(best_state)
    
    # Test evaluation
    full_train_set = defaultdict(set)
    for u, i in np.vstack([train_edges, val_edges]):
        full_train_set[int(u)].add(int(i))
    
    user_emb, movie_emb = get_gat_ngcf_embeddings(model, model_class, data_device, config)
    
    hr_test, ndcg_test = evaluate_embeddings(
        user_emb, movie_emb,
        test_edges,
        full_train_set,
        num_negatives=config.eval.num_negatives,
        K_list=tuple(config.eval.k_list),
        rng=np.random.RandomState(config.training.seed + 999)
    )
    
    results = {}
    results.update(hr_test)
    results.update(ndcg_test)
    
    return results, model


def evaluate_gat_ngcf_full_graph(model, model_class, data_device, val_edges,
                                  user_train_set, config, epoch, verbose=True):
    model.eval()
    with torch.no_grad():
        user_emb, movie_emb = get_gat_ngcf_embeddings(model, model_class, data_device, config)
        
        hr, ndcg = evaluate_embeddings(
            user_emb, movie_emb,
            val_edges,
            user_train_set,
            num_negatives=config.eval.num_negatives,
            K_list=tuple(config.eval.k_list),
            rng=np.random.RandomState(config.training.seed + epoch)
        )
        
        return ndcg["NDCG@10"]


def get_gat_ngcf_embeddings(model, model_class, data_device, config):
    device = next(model.parameters()).device
    num_users = data_device["user"].num_nodes
    num_movies = data_device["movie"].num_nodes
    
    # For GAT models
    if 'GAT' in model_class.__name__:
        if model_class in [GAT_concat, GAT_attention_guided]:
            user_emb = model.user_emb.weight
            if hasattr(model, 'user_feat_lin'):
                user_feat = F.relu(model.user_feat_lin(data_device["user"].x))
                if hasattr(model, 'user_combiner'):
                    user_x = model.user_combiner(torch.cat([user_emb, user_feat], dim=1))
                else:
                    user_x = user_emb + user_feat
            else:
                user_x = user_emb + model.user_lin(data_device["user"].x)
            
            x_dict = {
                "user": user_x,
                "movie": model.movie_emb.weight
            }
        else:
            # Linear and MTL
            x_dict = {
                "user": model.user_emb.weight + model.user_lin(data_device["user"].x),
                "movie": model.movie_emb.weight
            }
        
        final_x = model.gat(x_dict, data_device.edge_index_dict)
    
    # For NGCF models
    elif 'NGCF' in model_class.__name__:
        # Get initial embeddings
        if model_class in [NGCF_concat, NGCF_propagation_aware]:
            if hasattr(model, 'user_combiner'):
                user_emb = model.user_emb.weight
                user_feat = F.relu(model.user_feat_lin(data_device["user"].x))
                user_x = model.user_combiner(torch.cat([user_emb, user_feat], dim=1))
            else:
                user_x = model.user_emb.weight + model.user_lin(data_device["user"].x)
        else:
            user_x = model.user_emb.weight + model.user_lin(data_device["user"].x)
        
        movie_x = model.movie_emb.weight
        
        edge_index = data_device["user", "rates", "movie"].edge_index
        
        user_embeddings = [user_x]
        for conv in model.convs:
            user_x = conv(user_x, edge_index)
            user_embeddings.append(user_x)
        
        # Concatenate and project
        final_user = torch.cat(user_embeddings, dim=1)
        final_user = model.classifier_linear(final_user)
        
        return final_user.cpu().numpy(), movie_x.cpu().numpy()
    
    else:
        raise ValueError(f"Unknown model class: {model_class.__name__}")
    
    return final_x["user"].cpu().numpy(), final_x["movie"].cpu().numpy()