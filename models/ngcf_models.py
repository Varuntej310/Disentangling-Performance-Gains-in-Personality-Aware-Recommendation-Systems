"""
NGCF-based recommendation models with personality integration

Neural Graph Collaborative Filtering explicitly models high-order 
connectivity through embedding propagation, allowing personality 
to influence how collaborative signals are aggregated.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import HeteroData


class NGCFConv(torch.nn.Module):
    """
    NGCF convolution layer with message passing
    
    Explicitly models interaction between user/item embeddings during propagation
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.W1 = torch.nn.Linear(in_channels, out_channels)
        self.W2 = torch.nn.Linear(in_channels, out_channels)
    
    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        # Normalize by degree
        row, col = edge_index
        deg = torch.bincount(row, minlength=x.size(0)).float()
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        
        # Aggregate messages
        messages = torch.zeros_like(x)
        for i in range(edge_index.size(1)):
            src, dst = edge_index[0, i], edge_index[1, i]
            # Element-wise product (interaction)
            messages[dst] += norm[i] * (x[src] * x[dst])
        
        # Combine original embedding and messages
        out = self.W1(x) + self.W2(messages)
        return F.leaky_relu(out)


class NGCF(torch.nn.Module):
    """Base NGCF encoder with multiple layers"""
    def __init__(self, hidden_channels, num_layers=3):
        super().__init__()
        self.num_layers = num_layers
        
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(NGCFConv(hidden_channels, hidden_channels))
    
    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        embeddings = [x]
        
        for conv in self.convs:
            x = conv(x, edge_index)
            embeddings.append(x)
        
        # Concatenate all layer embeddings (like NGCF paper)
        return torch.cat(embeddings, dim=1)


class Classifier(torch.nn.Module):
    """Link prediction classifier"""
    def forward(self, x_user: Tensor, x_movie: Tensor, edge_label_index: Tensor) -> Tensor:
        edge_feat_user = x_user[edge_label_index[0]]
        edge_feat_movie = x_movie[edge_label_index[1]]
        return (edge_feat_user * edge_feat_movie).sum(dim=-1)


class PersonalityPredictor(torch.nn.Module):
    """Personality prediction head"""
    def __init__(self, in_channels, out_channels=5):
        super().__init__()
        self.lin1 = torch.nn.Linear(in_channels, in_channels // 2)
        self.lin2 = torch.nn.Linear(in_channels // 2, out_channels)
    
    def forward(self, x_user: Tensor) -> Tensor:
        x = F.relu(self.lin1(x_user))
        x = self.lin2(x)
        return x


class NGCF_linear(torch.nn.Module):
    """
    NGCF with linear personality integration
    
    Personality is linearly added to user embeddings before propagation.
    This affects how collaborative signals are aggregated across layers.
    """
    def __init__(self, hidden_channels, num_users, num_movies, num_layers=3):
        super().__init__()
        self.num_users = num_users
        self.num_movies = num_movies
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        
        self.user_lin = torch.nn.Linear(5, hidden_channels)
        self.user_emb = torch.nn.Embedding(num_users, hidden_channels)
        self.movie_emb = torch.nn.Embedding(num_movies, hidden_channels)
        
        # NGCF layers
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(NGCFConv(hidden_channels, hidden_channels))
        
        # Output dimension is hidden_channels * (num_layers + 1) due to concatenation
        output_dim = hidden_channels * (num_layers + 1)
        self.classifier_linear = torch.nn.Linear(output_dim, hidden_channels)
        self.classifier = Classifier()
        
        self.metadata = None
    
    def set_metadata(self, metadata):
        self.metadata = metadata
    
    def forward(self, data: HeteroData) -> Tensor:
        # Get batch node IDs
        user_ids = data["user"].n_id
        movie_ids = data["movie"].n_id
        
        # Initial embeddings with personality
        user_emb = self.user_emb(user_ids) + self.user_lin(data["user"].x)
        movie_emb = self.movie_emb(movie_ids)
        
        # Create bipartite graph representation
        # For simplicity, process user->movie edges
        edge_index = data["user", "rates", "movie"].edge_index
        
        # Store initial embeddings
        user_embeddings = [user_emb]
        movie_embeddings = [movie_emb]
        
        # Propagate through NGCF layers
        # Note: Simplified version - full NGCF would handle bipartite graph more carefully
        for conv in self.convs:
            # In practice, would alternate between user and movie updates
            # Here we use a simplified version
            user_emb_new = conv(user_emb, edge_index)
            user_embeddings.append(user_emb_new)
            user_emb = user_emb_new
        
        # Concatenate all layers
        final_user = torch.cat(user_embeddings, dim=1)
        final_user = self.classifier_linear(final_user)
        
        return self.classifier(
            final_user,
            movie_emb,
            data["user", "rates", "movie"].edge_label_index,
        )


class NGCF_concat(torch.nn.Module):
    """
    NGCF with concatenation personality integration
    
    Personality features are concatenated with embeddings and transformed
    before propagation through NGCF layers.
    """
    def __init__(self, hidden_channels, num_users, num_movies, 
                 user_feature_dim=5, num_layers=3):
        super().__init__()
        self.num_users = num_users
        self.num_movies = num_movies
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        
        self.user_emb = torch.nn.Embedding(num_users, hidden_channels)
        self.movie_emb = torch.nn.Embedding(num_movies, hidden_channels)
        
        # Personality transformation
        self.user_feat_lin = torch.nn.Linear(user_feature_dim, hidden_channels)
        self.user_combiner = torch.nn.Linear(hidden_channels * 2, hidden_channels)
        
        # NGCF layers
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(NGCFConv(hidden_channels, hidden_channels))
        
        output_dim = hidden_channels * (num_layers + 1)
        self.classifier_linear = torch.nn.Linear(output_dim, hidden_channels)
        self.classifier = Classifier()
        
        self.metadata = None
    
    def set_metadata(self, metadata):
        self.metadata = metadata
    
    def forward(self, data: HeteroData) -> Tensor:
        user_embedding = self.user_emb(data["user"].n_id)
        user_features = F.relu(self.user_feat_lin(data["user"].x))
        
        # Combine embeddings and personality
        combined_user_vec = torch.cat([user_embedding, user_features], dim=1)
        user_emb = self.user_combiner(combined_user_vec)
        movie_emb = self.movie_emb(data["movie"].n_id)
        
        edge_index = data["user", "rates", "movie"].edge_index
        
        # Propagate
        user_embeddings = [user_emb]
        for conv in self.convs:
            user_emb = conv(user_emb, edge_index)
            user_embeddings.append(user_emb)
        
        final_user = torch.cat(user_embeddings, dim=1)
        final_user = self.classifier_linear(final_user)
        
        return self.classifier(
            final_user,
            movie_emb,
            data["user", "rates", "movie"].edge_label_index,
        )


class NGCF_mtl(torch.nn.Module):
    """
    NGCF with multi-task learning
    
    Primary task: Link prediction
    Auxiliary task: Personality prediction from propagated embeddings
    
    The propagated embeddings capture high-order collaborative signals
    which may be predictive of personality traits.
    """
    def __init__(self, hidden_channels, num_users, num_movies, num_layers=3):
        super().__init__()
        self.num_users = num_users
        self.num_movies = num_movies
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        
        self.user_lin = torch.nn.Linear(5, hidden_channels)
        self.user_emb = torch.nn.Embedding(num_users, hidden_channels)
        self.movie_emb = torch.nn.Embedding(num_movies, hidden_channels)
        
        # NGCF layers
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(NGCFConv(hidden_channels, hidden_channels))
        
        output_dim = hidden_channels * (num_layers + 1)
        self.classifier_linear = torch.nn.Linear(output_dim, hidden_channels)
        self.classifier = Classifier()
        
        # Personality predictor uses final propagated embeddings
        self.personality_predictor = PersonalityPredictor(output_dim, out_channels=5)
        
        self.metadata = None
    
    def set_metadata(self, metadata):
        self.metadata = metadata
    
    def forward(self, data: HeteroData) -> dict:
        user_emb = self.user_emb(data["user"].n_id) + self.user_lin(data["user"].x)
        movie_emb = self.movie_emb(data["movie"].n_id)
        
        edge_index = data["user", "rates", "movie"].edge_index
        
        # Propagate
        user_embeddings = [user_emb]
        for conv in self.convs:
            user_emb = conv(user_emb, edge_index)
            user_embeddings.append(user_emb)
        
        # Concatenate all layers for predictions
        final_user_concat = torch.cat(user_embeddings, dim=1)
        
        # Link prediction
        final_user_link = self.classifier_linear(final_user_concat)
        pred_link = self.classifier(
            final_user_link,
            movie_emb,
            data["user", "rates", "movie"].edge_label_index,
        )
        
        # Personality prediction from propagated embeddings
        pred_personality = self.personality_predictor(final_user_concat)
        
        return {"link": pred_link, "personality": pred_personality}


class NGCF_propagation_aware(torch.nn.Module):
    """
    NGCF with personality-aware propagation
    
    Uses personality traits to modulate how embeddings are propagated,
    allowing different personality types to aggregate information differently.
    """
    def __init__(self, hidden_channels, num_users, num_movies, num_layers=3):
        super().__init__()
        self.num_users = num_users
        self.num_movies = num_movies
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        
        self.user_emb = torch.nn.Embedding(num_users, hidden_channels)
        self.movie_emb = torch.nn.Embedding(num_movies, hidden_channels)
        
        # Personality-based propagation weights
        self.personality_gates = torch.nn.ModuleList([
            torch.nn.Linear(5, 1) for _ in range(num_layers)
        ])
        
        self.user_lin = torch.nn.Linear(5, hidden_channels)
        
        # NGCF layers
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(NGCFConv(hidden_channels, hidden_channels))
        
        output_dim = hidden_channels * (num_layers + 1)
        self.classifier_linear = torch.nn.Linear(output_dim, hidden_channels)
        self.classifier = Classifier()
        
        self.metadata = None
    
    def set_metadata(self, metadata):
        self.metadata = metadata
    
    def forward(self, data: HeteroData) -> Tensor:
        user_emb = self.user_emb(data["user"].n_id) + self.user_lin(data["user"].x)
        movie_emb = self.movie_emb(data["movie"].n_id)
        
        edge_index = data["user", "rates", "movie"].edge_index
        
        # Compute personality-based gates for each layer
        personality_weights = [
            torch.sigmoid(gate(data["user"].x))
            for gate in self.personality_gates
        ]
        
        # Propagate with personality-modulated aggregation
        user_embeddings = [user_emb]
        for i, conv in enumerate(self.convs):
            propagated = conv(user_emb, edge_index)
            # Modulate by personality
            user_emb = personality_weights[i] * propagated + (1 - personality_weights[i]) * user_emb
            user_embeddings.append(user_emb)
        
        final_user = torch.cat(user_embeddings, dim=1)
        final_user = self.classifier_linear(final_user)
        
        return self.classifier(
            final_user,
            movie_emb,
            data["user", "rates", "movie"].edge_label_index,
        )