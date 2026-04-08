"""
Last.fm specific GNN models
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import SAGEConv, to_hetero
from torch_geometric.data import HeteroData


class UserFeatureEncoder(torch.nn.Module):
    """MLP to encode user features"""
    def __init__(self, in_dim, hidden_channels):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, hidden_channels),
            torch.nn.ReLU(),
        )
    
    def forward(self, x):
        return self.mlp(x)


class GNN(torch.nn.Module):
    """Base GNN encoder using GraphSAGE"""
    def __init__(self, hidden_channels):
        super().__init__()
        self.conv1 = SAGEConv(hidden_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
    
    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x


class Classifier(torch.nn.Module):
    """Link prediction classifier using dot product"""
    def forward(self, x_user: Tensor, x_artist: Tensor, edge_label_index: Tensor) -> Tensor:
        edge_feat_user = x_user[edge_label_index[0]]
        edge_feat_artist = x_artist[edge_label_index[1]]
        return (edge_feat_user * edge_feat_artist).sum(dim=-1)
    
class UserFeaturePredictor(torch.nn.Module):
    """Predict user features from embeddings (for MTL)"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.lin1 = torch.nn.Linear(in_channels, in_channels)
        self.lin2 = torch.nn.Linear(in_channels, out_channels)
    
    def forward(self, x_user: Tensor) -> Tensor:
        x = F.relu(self.lin1(x_user))
        x = self.lin2(x)
        return x


class Model_without(torch.nn.Module):
    """Baseline model without user features"""
    def __init__(self, hidden_channels, num_users, num_artists):
        super().__init__()
        self.user_emb = torch.nn.Embedding(num_users, hidden_channels)
        self.artist_emb = torch.nn.Embedding(num_artists, hidden_channels)
        self.gnn = GNN(hidden_channels)
        self.classifier = Classifier()
        
        self.metadata = None
    
    def set_metadata(self, metadata):
        self.metadata = metadata
        self.gnn = to_hetero(self.gnn, metadata=metadata)
    
    def forward(self, data: HeteroData) -> Tensor:
        user_ids = data["user"].n_id
        artist_ids = data["artist"].n_id
        
        x_dict = {
            "user": self.user_emb(user_ids),
            "artist": self.artist_emb(artist_ids),
        }
        
        x_dict = self.gnn(x_dict, data.edge_index_dict)
        
        return self.classifier(
            x_dict["user"],
            x_dict["artist"],
            data["user", "rates", "artist"].edge_label_index,
        )


class Model_userfeat_add(torch.nn.Module):
    """Model with user features added to embeddings"""
    def __init__(self, hidden_channels, num_users, num_artists, user_feat_dim):
        super().__init__()
        self.user_emb = torch.nn.Embedding(num_users, hidden_channels)
        self.artist_emb = torch.nn.Embedding(num_artists, hidden_channels)
        self.user_feat_encoder = UserFeatureEncoder(user_feat_dim, hidden_channels)
        self.gnn = GNN(hidden_channels)
        self.classifier = Classifier()
        
        self.metadata = None
    
    def set_metadata(self, metadata):
        self.metadata = metadata
        self.gnn = to_hetero(self.gnn, metadata=metadata)
    
    def forward(self, data: HeteroData) -> Tensor:
        user_ids = data["user"].n_id
        artist_ids = data["artist"].n_id
        
        user_feat_emb = self.user_feat_encoder(data["user"].x)
        
        x_dict = {
            "user": self.user_emb(user_ids) + user_feat_emb,
            "artist": self.artist_emb(artist_ids),
        }
        
        x_dict = self.gnn(x_dict, data.edge_index_dict)
        
        return self.classifier(
            x_dict["user"],
            x_dict["artist"],
            data["user", "rates", "artist"].edge_label_index,
        )


class Model_userfeat_concat(torch.nn.Module):
    """Model with user features concatenated to embeddings"""
    def __init__(self, hidden_channels, num_users, num_artists, user_feat_dim):
        super().__init__()
        self.user_emb = torch.nn.Embedding(num_users, hidden_channels)
        self.artist_emb = torch.nn.Embedding(num_artists, hidden_channels)
        self.user_feat_encoder = UserFeatureEncoder(user_feat_dim, hidden_channels)
        self.user_fusion = nn.Linear(2 * hidden_channels, hidden_channels)
        self.gnn = GNN(hidden_channels)
        self.classifier = Classifier()
        
        self.metadata = None
    
    def set_metadata(self, metadata):
        self.metadata = metadata
        self.gnn = to_hetero(self.gnn, metadata=metadata)
    
    def forward(self, data: HeteroData) -> Tensor:
        user_ids = data["user"].n_id
        artist_ids = data["artist"].n_id
        
        user_feat_emb = self.user_feat_encoder(data["user"].x)
        user_input = torch.cat([self.user_emb(user_ids), user_feat_emb], dim=-1)
        user_input = self.user_fusion(user_input)
        
        x_dict = {
            "user": user_input,
            "artist": self.artist_emb(artist_ids),
        }
        
        x_dict = self.gnn(x_dict, data.edge_index_dict)
        
        return self.classifier(
            x_dict["user"],
            x_dict["artist"],
            data["user", "rates", "artist"].edge_label_index,
        )
    

class Model_userfeat_mtl(torch.nn.Module):
    """Multi-task learning model with user feature prediction"""
    def __init__(self, hidden_channels, num_users, num_artists, user_feat_dim):
        super().__init__()
        self.user_emb = torch.nn.Embedding(num_users, hidden_channels)
        self.artist_emb = torch.nn.Embedding(num_artists, hidden_channels)
        self.user_feat_encoder = UserFeatureEncoder(user_feat_dim, hidden_channels)
        self.gnn = GNN(hidden_channels)
        self.classifier = Classifier()
        self.feature_predictor = UserFeaturePredictor(hidden_channels, user_feat_dim)
        
        self.metadata = None
    
    def set_metadata(self, metadata):
        self.metadata = metadata
        self.gnn = to_hetero(self.gnn, metadata=metadata)
    
    def forward(self, data: HeteroData) -> dict:
        user_ids = data["user"].n_id
        artist_ids = data["artist"].n_id
        
        user_feat_emb = self.user_feat_encoder(data["user"].x)
        
        x_dict = {
            "user": self.user_emb(user_ids) + user_feat_emb,
            "artist": self.artist_emb(artist_ids),
        }
        
        final_x_dict = self.gnn(x_dict, data.edge_index_dict)
        
        pred_link = self.classifier(
            final_x_dict["user"],
            final_x_dict["artist"],
            data["user", "rates", "artist"].edge_label_index,
        )
        
        pred_features = self.feature_predictor(final_x_dict["user"])
        
        return {
            "link": pred_link,
            "features": pred_features
        }