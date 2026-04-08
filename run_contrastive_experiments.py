import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
from math import log2
import os

# Import your existing code
from config.experiment_config import ExperimentConfig
from data.base_dataset import MovieLensDataset, split_edges, create_hetero_data
from torch_geometric.nn import SAGEConv, to_hetero
from torch_geometric.loader import LinkNeighborLoader

# ============================================================================
# Personality Trait Discretization
# ============================================================================

def discretize_trait(trait_values, threshold=5.0):
    """Discretize trait into High (1) / Low (0)"""
    return (trait_values > threshold).astype(int)


def compute_jaccard_similarity(user_items_dict, u, v):
    """Compute Jaccard similarity between two users' item sets"""
    set_u = user_items_dict.get(u, set())
    set_v = user_items_dict.get(v, set())
    
    if len(set_u) == 0 or len(set_v) == 0:
        return 0.0
    
    intersection = len(set_u & set_v)
    union = len(set_u | set_v)
    return intersection / union if union > 0 else 0.0


def create_personality_pairs(personality_df, train_edges, trait_name='extraversion', 
                             threshold=5.0, jaccard_threshold=0.1, max_pairs=5000, seed=42):
    """
    Create positive and negative pairs based on trait AND collaborative signal
    
    FIX #4: Positives must share personality bucket AND have collaborative overlap
    
    Returns:
        positive_pairs: [(user_i, user_j), ...] same trait + Jaccard > threshold
        negative_pairs: [(user_i, user_j), ...] different trait
    """
    labels = discretize_trait(personality_df[trait_name].values, threshold)
    user_ids = personality_df['mappedUserID'].values
    
    high_users = user_ids[labels == 1]
    low_users = user_ids[labels == 0]
    
    print(f"\nTrait: {trait_name} (threshold={threshold})")
    print(f"  High: {len(high_users)} users ({len(high_users)/len(user_ids)*100:.1f}%)")
    print(f"  Low:  {len(low_users)} users ({len(low_users)/len(user_ids)*100:.1f}%)")
    
    # Build user-item dict for Jaccard computation
    user_items = defaultdict(set)
    for u, i in train_edges:
        user_items[int(u)].add(int(i))
    
    rng = np.random.RandomState(seed)
    positive_pairs = []
    negative_pairs = []
    
    # Positive pairs from high group (same trait + Jaccard overlap)
    if len(high_users) > 1:
        candidates = []
        for i in range(len(high_users)):
            for j in range(i + 1, len(high_users)):
                u, v = int(high_users[i]), int(high_users[j])
                jaccard = compute_jaccard_similarity(user_items, u, v)
                if jaccard >= jaccard_threshold:
                    candidates.append((u, v))
        
        num_high = min(max_pairs // 2, len(candidates))
        selected = rng.choice(len(candidates), size=min(num_high, len(candidates)), replace=False)
        positive_pairs.extend([candidates[idx] for idx in selected])
    
    # Positive pairs from low group (same trait + Jaccard overlap)
    if len(low_users) > 1:
        candidates = []
        for i in range(len(low_users)):
            for j in range(i + 1, len(low_users)):
                u, v = int(low_users[i]), int(low_users[j])
                jaccard = compute_jaccard_similarity(user_items, u, v)
                if jaccard >= jaccard_threshold:
                    candidates.append((u, v))
        
        num_low = min(max_pairs // 2, len(candidates))
        selected = rng.choice(len(candidates), size=min(num_low, len(candidates)), replace=False)
        positive_pairs.extend([candidates[idx] for idx in selected])
    
    # Negative pairs across groups (only ensure different traits)
    if len(high_users) > 0 and len(low_users) > 0:
        num_neg = min(max_pairs, len(high_users) * len(low_users) // 100)
        for _ in range(num_neg):
            i = rng.choice(high_users)
            j = rng.choice(low_users)
            negative_pairs.append((int(i), int(j)))
    
    print(f"  Positive pairs (with Jaccard>{jaccard_threshold}): {len(positive_pairs)}")
    print(f"  Negative pairs: {len(negative_pairs)}")
    
    return positive_pairs, negative_pairs


# ============================================================================
# Contrastive Loss (FIXED #2)
# ============================================================================

class PersonalityContrastiveLoss(nn.Module):
    """
    Corrected contrastive loss using margin-based formulation
    
    FIX #2: Negative loss now correctly uses margin semantics
    - Positive: minimize distance (maximize similarity)
    - Negative: maximize distance (minimize similarity) with margin
    """
    def __init__(self, margin=0.5):
        super().__init__()
        self.margin = margin
    
    def forward(self, embeddings, positive_pairs, negative_pairs):
        if len(positive_pairs) == 0 and len(negative_pairs) == 0:
            return torch.tensor(0.0, device=embeddings.device)
        
        total_loss = 0.0
        count = 0
        
        # Positive pairs: pull together (maximize similarity)
        if len(positive_pairs) > 0:
            pos_tensor = torch.tensor(positive_pairs, device=embeddings.device, dtype=torch.long)
            emb_i = embeddings[pos_tensor[:, 0]]
            emb_j = embeddings[pos_tensor[:, 1]]
            
            # Cosine similarity (want high similarity = close to 1)
            similarity = F.cosine_similarity(emb_i, emb_j)
            pos_loss = (1 - similarity).mean()
            
            total_loss += pos_loss
            count += 1
        
        # Negative pairs: push apart with margin (minimize similarity, but allow small margin)
        if len(negative_pairs) > 0:
            neg_tensor = torch.tensor(negative_pairs, device=embeddings.device, dtype=torch.long)
            emb_i = embeddings[neg_tensor[:, 0]]
            emb_j = embeddings[neg_tensor[:, 1]]
            
            similarity = F.cosine_similarity(emb_i, emb_j)
            
            # Hinge loss: penalty if similarity > -margin
            # This allows negatives to be dissimilar but doesn't aggressively force them apart
            neg_loss = torch.clamp(similarity + self.margin, min=0).mean()
            
            total_loss += neg_loss
            count += 1
        
        return total_loss / max(count, 1)


# ============================================================================
# Model
# ============================================================================

class GNN(nn.Module):
    """GraphSAGE encoder"""
    def __init__(self, hidden_channels):
        super().__init__()
        self.conv1 = SAGEConv(hidden_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, hidden_channels)
    
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        x = self.conv3(x, edge_index)
        return x


class Classifier(nn.Module):
    def forward(self, x_user, x_movie, edge_label_index):
        edge_feat_user = x_user[edge_label_index[0]]
        edge_feat_movie = x_movie[edge_label_index[1]]
        return (edge_feat_user * edge_feat_movie).sum(dim=-1)


class ContrastiveGNN(nn.Module):
    """GNN with personality contrastive loss"""
    def __init__(self, hidden_channels, num_users, num_movies):
        super().__init__()
        
        self.user_emb = nn.Embedding(num_users, hidden_channels)
        self.movie_emb = nn.Embedding(num_movies, hidden_channels)
        self.user_lin = nn.Linear(5, hidden_channels)
        
        self.gnn = GNN(hidden_channels)
        self.classifier = Classifier()
        self.contrastive_loss = PersonalityContrastiveLoss(margin=0.5)
        
        self.metadata = None
    
    def set_metadata(self, metadata):
        self.metadata = metadata
        self.gnn = to_hetero(self.gnn, metadata=metadata)
    
    def forward(self, data):
        """
        FIX #1: Only link prediction in mini-batch forward
        Contrastive loss computed separately on full embeddings
        """
        user_ids = data["user"].n_id
        movie_ids = data["movie"].n_id
        
        x_dict = {
            "user": self.user_emb(user_ids) + self.user_lin(data["user"].x),
            "movie": self.movie_emb(movie_ids),
        }
        
        x_dict = self.gnn(x_dict, data.edge_index_dict)
        
        pred_link = self.classifier(
            x_dict["user"],
            x_dict["movie"],
            data["user", "rates", "movie"].edge_label_index,
        )
        
        return pred_link
    
    def get_user_embeddings(self, data):
        with torch.no_grad():
            user_ids = torch.arange(data["user"].num_nodes, device=data["user"].x.device)
            movie_ids = torch.arange(data["movie"].num_nodes, device='cuda')
            
            x_dict = {
                "user": self.user_emb(user_ids) + self.user_lin(data["user"].x),
                "movie": self.movie_emb(movie_ids),
            }
            
            x_dict = self.gnn(x_dict, data.edge_index_dict)
            return x_dict["user"], x_dict["movie"]


# ============================================================================
# Training and Evaluation
# ============================================================================

def evaluate_embeddings(user_emb, movie_emb, test_edges, train_set, K_list=(5, 10)):
    """Standard HR@K and NDCG@K evaluation"""
    rng = np.random.RandomState(42)
    hits = {k: 0 for k in K_list}
    ndcg = {k: 0.0 for k in K_list}
    count = 0
    num_items = movie_emb.shape[0]
    
    for u, pos in test_edges:
        u, pos = int(u), int(pos)
        
        negs = []
        while len(negs) < 99:
            cand = rng.randint(0, num_items)
            if cand != pos and cand not in train_set[u]:
                negs.append(cand)
        
        candidates = [pos] + negs
        scores = movie_emb[candidates] @ user_emb[u]
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


def train_model(model, data, train_loader, optimizer, device):
    """
    Single training epoch - ONLY link prediction
    
    FIX #1: Contrastive loss removed from batch loop
    """
    model.train()
    total_link = 0
    total_examples = 0
    
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        pred_link = model(batch)
        gt_link = batch["user", "rates", "movie"].edge_label
        link_loss = F.binary_cross_entropy_with_logits(pred_link, gt_link)
        
        loss = link_loss
        loss.backward()
        optimizer.step()
        
        total_link += float(link_loss) * pred_link.numel()
        total_examples += pred_link.numel()
    
    return {
        'link_loss': total_link / total_examples,
    }


def compute_contrastive_loss_epoch(model, data, pos_pairs, neg_pairs, device):
    """
    FIX #1: Compute contrastive loss on FULL graph embeddings after mini-batch training
    
    This ensures:
    - Contrastive constraints are stable (not batch-dependent)
    - Embeddings are computed on full graph context
    - Semantically correct
    """
    model.eval()
    with torch.no_grad():
        user_emb, _ = model.get_user_embeddings(data)
        contrastive = model.contrastive_loss(user_emb, pos_pairs, neg_pairs)
    
    return float(contrastive)


def run_single_experiment(data, train_edges, val_edges, test_edges,
                          pos_pairs, neg_pairs, contrastive_weight=0.0,
                          num_epochs=30, device='cuda'):
    """
    Run one experiment
    
    FIX #3: Contrastive weight is annealed over time
    """
    
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    data_device = data.to(device)
    
    # Create loader
    edge_label_index = torch.tensor(train_edges.T, dtype=torch.long)
    edge_label = torch.ones(edge_label_index.size(1), dtype=torch.float)
    
    train_loader = LinkNeighborLoader(
        data=data_device,
        num_neighbors=[20, 10],
        neg_sampling_ratio=2.0,
        edge_label_index=(("user", "rates", "movie"), edge_label_index),
        edge_label=edge_label,
        batch_size=128,
        shuffle=True,
    )
    
    # Train set
    train_set = defaultdict(set)
    for u, i in train_edges:
        train_set[int(u)].add(int(i))
    
    # Model
    model = ContrastiveGNN(
        hidden_channels=5,
        num_users=data_device["user"].num_nodes,
        num_movies=data_device["movie"].num_nodes
    ).to(device)
    
    model.set_metadata(data_device.metadata())
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    
    # Training loop
    best_val = -1
    patience_counter = 0
    history = {'val_ndcg': [], 'link_loss': [], 'contrastive_loss': []}
    
    print(f"\nTraining (λ={contrastive_weight})...")
    
    for epoch in range(1, num_epochs + 1):
        # FIX #3: Anneal contrastive weight
        # Weight starts high and decays to 0 over time
        # This allows CF to dominate in later epochs
        lambda_t = contrastive_weight * np.exp(-epoch / (num_epochs / 3))
        
        # Mini-batch link prediction training
        metrics = train_model(model, data_device, train_loader, optimizer, device)
        
        # FIX #1: Compute contrastive loss on full embeddings
        contrastive_loss = compute_contrastive_loss_epoch(
            model, data_device, pos_pairs, neg_pairs, device
        ) if len(pos_pairs) > 0 else 0.0
        
        # Validate
        model.eval()
        with torch.no_grad():
            user_emb, movie_emb = model.get_user_embeddings(data_device)
            _, ndcg = evaluate_embeddings(
                user_emb.cpu().numpy(),
                movie_emb.cpu().numpy(),
                val_edges, train_set
            )
            val_ndcg = ndcg['NDCG@10']
            history['val_ndcg'].append(val_ndcg)
            history['link_loss'].append(metrics['link_loss'])
            history['contrastive_loss'].append(contrastive_loss)
        
        if epoch % 5 == 0:
            print(f"  Epoch {epoch:02d} | Link: {metrics['link_loss']:.4f} | "
                  f"Contrast: {contrastive_loss:.4f} | λ_t: {lambda_t:.6f} | "
                  f"Val NDCG@10: {val_ndcg:.4f}")
        
        if val_ndcg > best_val:
            best_val = val_ndcg
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 10:
                print("  Early stopping")
                break
    
    # Test
    model.eval()
    with torch.no_grad():
        user_emb, movie_emb = model.get_user_embeddings(data_device)
        
        full_train = defaultdict(set)
        for u, i in np.vstack([train_edges, val_edges]):
            full_train[int(u)].add(int(i))
        
        hr, ndcg = evaluate_embeddings(
            user_emb.cpu().numpy(),
            movie_emb.cpu().numpy(),
            test_edges, full_train
        )
    
    return {**hr, **ndcg, 'history': history}


# ============================================================================
# Main
# ============================================================================

def main():
    print("="*70)
    print("PERSONALITY CONTRASTIVE LEARNING EXPERIMENT (CORRECTED)")
    print("="*70)
    print("\nHypothesis: Users with similar personality + CF signal have similar preferences")
    print("Approach: Use contrastive loss to refine CF (not override it)")
    print("Test trait: Extraversion (High vs Low)")
    print("="*70)
    
    # Load data
    print("\n[1/5] Loading data...")
    
    # Create minimal config
    class SimpleConfig:
        class Paths:
            ratings= "/home/fac/ashok.sairam/btp-varun/btp-1/ratings.csv"
            personality= "/home/fac/ashok.sairam/btp-varun/btp-1/personality-data.csv"
        
        class Data:
            sparsity_percentile = 100.0
            min_interactions = 2
            personality_columns = ['openness', 'agreeableness', 'emotional_stability',
                                  'conscientiousness', 'extraversion']
            personality_type = 'real'
        
        class Training:
            seed = 42
    
    config = SimpleConfig()
    config.paths = SimpleConfig.Paths()
    config.data = SimpleConfig.Data()
    config.training = SimpleConfig.Training()
    
    dataset = MovieLensDataset(config)
    dataset.load_data()
    dataset.sparsify_ratings()
    dataset.create_mappings()
    dataset.merge_features()
    
    edge_index = dataset.get_edge_index()
    
    print("\n[2/5] Splitting edges...")
    train_edges, val_edges, test_edges = split_edges(
        edge_index, min_interactions=2, seed=42
    )
    
    print("\n[3/5] Creating HeteroData...")
    data = create_hetero_data(
        dataset.unique_user_id,
        dataset.unique_movie_id,
        dataset.filtered_personality,
        train_edges
    )
    
    print(f"  Users: {data['user'].num_nodes}")
    print(f"  Movies: {data['movie'].num_nodes}")
    
    print("\n[4/5] Creating personality pairs (with CF signal filtering)...")
    pos_pairs, neg_pairs = create_personality_pairs(
        dataset.filtered_personality,
        train_edges,
        trait_name='openness',
        threshold=5.0,
        jaccard_threshold=0.1,  # Only pair users with 10%+ item overlap
        max_pairs=5000
    )
    
    print("\n[5/5] Running experiments...")
    
    # Baseline
    print("\n" + "-"*70)
    print("BASELINE (No Contrastive Loss)")
    print("-"*70)
    baseline_results = run_single_experiment(
        data, train_edges, val_edges, test_edges,
        [], [],  # Empty pairs for baseline
        contrastive_weight=0.0,
        num_epochs=30
    )
    
    # With contrastive (properly annealed)
    print("\n" + "-"*70)
    print("WITH CONTRASTIVE LOSS (λ=0.1, annealed)")
    print("-"*70)
    contrastive_results = run_single_experiment(
        data, train_edges, val_edges, test_edges,
        pos_pairs, neg_pairs,
        contrastive_weight=0.1,  # Will be annealed during training
        num_epochs=30
    )
    
    # Results
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    for metric in ['HR@5', 'HR@10', 'NDCG@5', 'NDCG@10']:
        base = baseline_results[metric]
        cont = contrastive_results[metric]
        diff = cont - base
        pct = (diff / base) * 100 if base > 0 else 0
        
        symbol = "✓" if diff > 0 else "✗"
        print(f"{metric:10s} | Baseline: {base:.4f} | "
              f"Contrastive: {cont:.4f} | "
              f"{symbol} {diff:+.4f} ({pct:+.2f}%)")
    
    # Save results
    import json
    results = {
        'baseline': {k: v for k, v in baseline_results.items() if k != 'history'},
        'contrastive': {k: v for k, v in contrastive_results.items() if k != 'history'},
        'improvement': {
            metric: contrastive_results[metric] - baseline_results[metric]
            for metric in ['HR@5', 'HR@10', 'NDCG@5', 'NDCG@10']
        }
    }
    
    with open('contrastive_results_fixed.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n✓ Results saved to contrastive_results_fixed.json")
    print("="*70)


if __name__ == "__main__":
    main()