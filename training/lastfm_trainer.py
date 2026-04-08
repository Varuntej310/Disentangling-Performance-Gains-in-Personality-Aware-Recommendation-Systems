"""
Training loops for Last.fm models (including MTL)
"""
import torch
import torch.nn.functional as F
import numpy as np
import copy
from collections import defaultdict
from torch_geometric.loader import LinkNeighborLoader

from training.evaluator import evaluate_embeddings


def train_lastfm_model(
    model_class,
    data,
    train_edges,
    val_edges,
    test_edges,
    config,
    user_feat_dim=None,
    verbose=True
):
    """
    Train a Last.fm GNN model (including MTL)
    
    Args:
        model_class: Model class to instantiate
        data: HeteroData object
        train_edges: Training edges
        val_edges: Validation edges
        test_edges: Test edges
        config: ExperimentConfig object
        user_feat_dim: Dimension of user features (required for add/concat/mtl models)
        verbose: Whether to print progress
    
    Returns:
        results: Dict of test metrics
        model: Trained model
    """
    torch.manual_seed(config.training.seed)
    np.random.seed(config.training.seed)
    
    device = torch.device(config.training.device if torch.cuda.is_available() else 'cpu')
    data_device = data.to(device)
    
    # Create train loader
    edge_label_index = torch.tensor(train_edges.T, dtype=torch.long)
    edge_label = torch.ones(edge_label_index.size(1), dtype=torch.float)
    
    train_loader = LinkNeighborLoader(
        data=data_device,
        num_neighbors=config.model.num_neighbors,
        neg_sampling_ratio=config.training.neg_sampling_ratio,
        edge_label_index=(("user", "rates", "artist"), edge_label_index),
        edge_label=edge_label,
        batch_size=config.training.batch_size,
        shuffle=True,
    )
    
    # Build user train set for evaluation
    user_train_set = defaultdict(set)
    for u, i in train_edges:
        user_train_set[int(u)].add(int(i))
    
    # Initialize model
    num_users = data_device["user"].num_nodes
    num_artists = data_device["artist"].num_nodes
    
    # Model instantiation depends on type
    if user_feat_dim is not None:
        model = model_class(
            hidden_channels=config.model.hidden_channels,
            num_users=num_users,
            num_artists=num_artists,
            user_feat_dim=user_feat_dim
        ).to(device)
    else:
        model = model_class(
            hidden_channels=config.model.hidden_channels,
            num_users=num_users,
            num_artists=num_artists
        ).to(device)
    
    # Set metadata for heterogeneous GNN
    model.set_metadata(data_device.metadata())
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config.training.lr)
    
    best_val_score = -np.inf
    best_state = None
    epochs_no_improve = 0
    
    # Get feature loss weight (for MTL)
    feature_loss_weight = getattr(config.model, 'feature_loss_weight', 0.5)
    
    # Training loop
    for epoch in range(1, config.training.num_epochs + 1):
        model.train()
        total_loss = 0.0
        total_examples = 0
        total_link_loss = 0.0
        total_feat_loss = 0.0
        
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            out = model(batch)
            
            # Check if MTL model (returns dict)
            if isinstance(out, dict):
                pred_link = out["link"]
                pred_features = out["features"]
                
                gt_link = batch["user", "rates", "artist"].edge_label
                user_ids = batch["user"].n_id
                gt_features = data_device["user"].x[user_ids]
                
                loss_link = F.binary_cross_entropy_with_logits(pred_link, gt_link)
                loss_feat = F.mse_loss(pred_features, gt_features)
                loss = loss_link + feature_loss_weight * loss_feat
                
                total_link_loss += float(loss_link) * pred_link.numel()
                total_feat_loss += float(loss_feat) * pred_features.shape[0]
            else:
                pred_link = out
                gt_link = batch["user", "rates", "artist"].edge_label
                loss = F.binary_cross_entropy_with_logits(pred_link, gt_link)
            
            loss.backward()
            optimizer.step()
            
            total_loss += float(loss) * gt_link.numel()
            total_examples += gt_link.numel()
        
        avg_loss = total_loss / total_examples
        
        if verbose:
            if isinstance(out, dict):
                print(f"Epoch {epoch:03d} | Loss {avg_loss:.4f} "
                      f"(Link: {total_link_loss/total_examples:.4f}, "
                      f"Feat: {total_feat_loss/total_examples:.4f})", end="")
            else:
                print(f"Epoch {epoch:03d} | Train Loss {avg_loss:.4f}", end="")
        
        # Validation evaluation
        val_score = evaluate_lastfm_model(
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
    
    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
    
    # Test evaluation
    full_train_set = defaultdict(set)
    for u, i in np.vstack([train_edges, val_edges]):
        full_train_set[int(u)].add(int(i))
    
    user_emb, artist_emb = get_full_embeddings_lastfm(model, model_class, data_device)
    
    hr_test, ndcg_test = evaluate_embeddings(
        user_emb, artist_emb,
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


def evaluate_lastfm_model(model, model_class, data_device, val_edges, user_train_set, 
                          config, epoch, verbose=True):
    """Evaluate Last.fm model on validation set"""
    model.eval()
    with torch.no_grad():
        user_emb, artist_emb = get_full_embeddings_lastfm(model, model_class, data_device)
        
        hr, ndcg = evaluate_embeddings(
            user_emb, artist_emb,
            val_edges,
            user_train_set,
            num_negatives=config.eval.num_negatives,
            K_list=tuple(config.eval.k_list),
            rng=np.random.RandomState(config.training.seed + epoch)
        )
        
        val_score = ndcg["NDCG@10"]
        return val_score


def get_full_embeddings_lastfm(model, model_class, data_device):
    """Get embeddings for all users and artists"""
    device = next(model.parameters()).device
    num_users = data_device["user"].num_nodes
    num_artists = data_device["artist"].num_nodes
    
    user_ids = torch.arange(num_users, device=device)
    artist_ids = torch.arange(num_artists, device=device)
    
    # Build user representation depending on model type
    if hasattr(model, "user_feat_encoder"):
        user_feat_emb = model.user_feat_encoder(data_device["user"].x)
        
        if hasattr(model, "user_fusion"):  # Concat model
            user_input = torch.cat(
                [model.user_emb(user_ids), user_feat_emb],
                dim=-1
            )
            user_input = model.user_fusion(user_input)
        else:  # Add or MTL model
            user_input = model.user_emb(user_ids) + user_feat_emb
    else:  # Without model
        user_input = model.user_emb(user_ids)
    
    x_dict = {
        "user": user_input,
        "artist": model.artist_emb(artist_ids),
    }
    
    final_x = model.gnn(x_dict, data_device.edge_index_dict)
    
    return final_x["user"].detach().cpu().numpy(), final_x["artist"].detach().cpu().numpy()