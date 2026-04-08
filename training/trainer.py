"""
Training loops updated for YAML config
"""
from logging import config
import torch
import torch.nn.functional as F
import numpy as np
import copy
from collections import defaultdict
from torch_geometric.loader import LinkNeighborLoader

from .evaluator import evaluate_embeddings
from models.graphsage_models import Model_concat, Model_graphsage
from models.lightgcn import bpr_loss


def train_gnn_model(
    model_class,
    data,
    train_edges,
    val_edges,
    test_edges,
    config,
    personality_loss_weight=0.0,
    verbose=True
):
    """
    Train a GNN-based recommendation model
    
    Args:
        model_class: Model class to instantiate
        data: HeteroData object
        train_edges: Training edges
        val_edges: Validation edges
        test_edges: Test edges
        config: ExperimentConfig object
        personality_loss_weight: Override config value if provided
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
        edge_label_index=(("user", "rates", "movie"), edge_label_index),
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
    num_movies = data_device["movie"].num_nodes
    
    model = model_class(
        hidden_channels=config.model.hidden_channels,
        num_users=num_users,
        num_movies=num_movies
    ).to(device)
    
    # Set metadata for heterogeneous GNN
    model.set_metadata(data_device.metadata())
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config.training.lr)
    # embed_params = list(model.user_emb.parameters()) + \
    #            list(model.movie_emb.parameters())

    # embed_param_ids = {id(p) for p in embed_params}

    # other_params = [
    #     p for p in model.parameters()
    #     if id(p) not in embed_param_ids
    # ]

    # optimizer = torch.optim.Adam(
    #     [
    #         {"params": embed_params, "weight_decay": 1e-4},
    #         {"params": other_params, "weight_decay": 0.0},
    #     ],
    #     lr=config.training.lr
    # )

    
    best_val_score = -np.inf
    best_state = None
    epochs_no_improve = 0
    
    # Use personality loss weight from argument or config
    if personality_loss_weight == 0.0 and hasattr(config.model, 'personality_loss_weight'):
        personality_loss_weight = config.model.personality_loss_weight

    # lambda_0 = personality_loss_weight
    # tau = config.training.num_epochs / 3

    # Training loop
    for epoch in range(1, config.training.num_epochs + 1):
        model.train()
        total_loss = 0.0
        total_examples = 0
        # lambda_t = lambda_0 * np.exp(-(epoch - 1) / tau)

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
                loss_pers = F.mse_loss(pred_personality, gt_personality) / gt_personality.var()
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
        
        # Validation evaluation
        val_score = evaluate_model_full_graph(
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
    
    user_emb, movie_emb = get_full_embeddings(model, model_class, data_device, config)
    
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


def evaluate_model_full_graph(model, model_class, data_device, val_edges, 
                               user_train_set, config, epoch, verbose=True):
    """Evaluate model on full graph"""
    model.eval()
    with torch.no_grad():
        user_emb, movie_emb = get_full_embeddings(model, model_class, data_device, config)
        
        hr, ndcg = evaluate_embeddings(
            user_emb, movie_emb,
            val_edges,
            user_train_set,
            num_negatives=config.eval.num_negatives,
            K_list=tuple(config.eval.k_list),
            rng=np.random.RandomState(config.training.seed + epoch)
        )
        
        val_score = ndcg["NDCG@10"]
        return val_score


def get_full_embeddings(model, model_class, data_device, config):
    """Get embeddings for all users and movies"""
    device = next(model.parameters()).device
    num_users = data_device["user"].num_nodes
    num_movies = data_device["movie"].num_nodes
    
    user_ids = torch.arange(num_users, device=device)
    movie_ids = torch.arange(num_movies, device=device)
    
    if model_class == Model_concat:
        user_emb = model.user_emb(user_ids)
        user_feat = F.relu(model.user_feat_lin(data_device["user"].x))
        user_x = model.user_combiner(torch.cat([user_emb, user_feat], dim=1))
        x_dict = {
            "user": user_x,
            "movie": model.movie_emb(movie_ids)
        }
    elif model_class == Model_graphsage:
        # graphsage model without any auxiliary features
        x_dict = {
            "user": model.user_emb(user_ids),
            "movie": model.movie_emb(movie_ids)
        }
    else:
        x_dict = {
            "user": model.user_emb(user_ids) + model.user_lin(data_device["user"].x),
            "movie": model.movie_emb(movie_ids)
        }
    
    final_x = model.gnn(x_dict, data_device.edge_index_dict)
    
    return final_x["user"].detach().cpu().numpy(), final_x["movie"].detach().cpu().numpy()


def train_lightgcn(data, train_edges, val_edges, test_edges, config, verbose=True):
    """Train LightGCN model"""
    from models.lightgcn import LightGCN
    
    torch.manual_seed(config.training.seed)
    np.random.seed(config.training.seed)
    
    device = torch.device(config.training.device if torch.cuda.is_available() else 'cpu')
    data_device = data.to(device)
    
    num_users = data_device["user"].num_nodes
    num_movies = data_device["movie"].num_nodes
    
    model = LightGCN(
        num_users, 
        num_movies, 
        config.model.embedding_dim, 
        config.model.num_layers
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config.training.lr)
    
    edge_index = data_device["user", "rates", "movie"].edge_index
    
    # Build train positives
    user_pos = defaultdict(set)
    for u, i in train_edges:
        user_pos[int(u)].add(int(i))
    
    best_val = -np.inf
    best_state = None
    no_improve = 0
    
    for epoch in range(1, config.training.num_epochs + 1):
        model.train()
        
        # Sample negatives
        users, pos_items, neg_items = [], [], []
        rng = np.random.RandomState(config.training.seed + epoch)
        
        for u, i in train_edges:
            u = int(u)
            i = int(i)
            neg = rng.randint(0, num_movies)
            while neg in user_pos[u]:
                neg = rng.randint(0, num_movies)
            
            users.append(u)
            pos_items.append(i)
            neg_items.append(neg)
        
        users = torch.tensor(users, device=device)
        pos_items = torch.tensor(pos_items, device=device)
        neg_items = torch.tensor(neg_items, device=device)
        
        user_emb, movie_emb = model(edge_index)
        
        loss = bpr_loss(
            user_emb[users],
            movie_emb[pos_items],
            movie_emb[neg_items]
        )
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            user_emb, movie_emb = model(edge_index)
            
            train_set = defaultdict(set)
            for u, i in train_edges:
                train_set[int(u)].add(int(i))
            
            _, ndcg = evaluate_embeddings(
                user_emb.cpu().numpy(),
                movie_emb.cpu().numpy(),
                val_edges,
                train_set,
                num_negatives=config.eval.num_negatives,
                K_list=tuple(config.eval.k_list),
                rng=np.random.RandomState(config.training.seed + 999)
            )
            
            val_score = ndcg["NDCG@10"]
        
        
        if val_score > best_val + 1e-6:
            best_val = val_score
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
            if verbose:
                print(f"Epoch {epoch:03d} | Loss {loss.item():.4f} | Val NDCG@10 {val_score:.4f}" "✓") 
        else:
            no_improve += 1
            if verbose:
                print(f"Epoch {epoch:03d} | Loss {loss.item():.4f} | Val NDCG@10 {val_score:.4f} (no improve: {no_improve})")
            if no_improve >= config.training.patience:
                if verbose:
                    print("Early stopping triggered.")
                break
    
    model.load_state_dict(best_state)
    
    # Test evaluation
    model.eval()
    with torch.no_grad():
        user_emb, movie_emb = model(edge_index)
        
        full_train = defaultdict(set)
        for u, i in np.vstack([train_edges, val_edges]):
            full_train[int(u)].add(int(i))
        
        hr_test, ndcg_test = evaluate_embeddings(
            user_emb.cpu().numpy(),
            movie_emb.cpu().numpy(),
            test_edges,
            full_train,
            num_negatives=config.eval.num_negatives,
            K_list=tuple(config.eval.k_list),
            rng=np.random.RandomState(config.training.seed + 1234)
        )
    
    results = {}
    results.update(hr_test)
    results.update(ndcg_test)
    
    return results, model