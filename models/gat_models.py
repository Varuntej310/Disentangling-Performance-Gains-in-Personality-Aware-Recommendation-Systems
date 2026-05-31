"""
GAT-based recommendation models with personality integration
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import GATConv, to_hetero
from torch_geometric.data import HeteroData


class GAT(torch.nn.Module):
    def __init__(self, hidden_channels, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        
        self.conv1 = GATConv(hidden_channels, hidden_channels // num_heads, 
                            heads=num_heads, dropout=0.1)
        self.conv2 = GATConv(hidden_channels, hidden_channels // num_heads, 
                            heads=num_heads, dropout=0.1)
        self.conv3 = GATConv(hidden_channels, hidden_channels // num_heads, 
                            heads=num_heads, dropout=0.1, concat=False)
        
    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.1, training=self.training)
        x = F.elu(self.conv2(x, edge_index))
        x = F.dropout(x, p=0.1, training=self.training)
        x = self.conv3(x, edge_index)
        return x


class Classifier(torch.nn.Module):
    def forward(self, x_user: Tensor, x_movie: Tensor, edge_label_index: Tensor) -> Tensor:
        edge_feat_user = x_user[edge_label_index[0]]
        edge_feat_movie = x_movie[edge_label_index[1]]
        return (edge_feat_user * edge_feat_movie).sum(dim=-1)


class PersonalityPredictor(torch.nn.Module):
    def __init__(self, in_channels, out_channels=5):
        super().__init__()
        self.lin1 = torch.nn.Linear(in_channels, in_channels)
        self.lin2 = torch.nn.Linear(in_channels, out_channels)
    
    def forward(self, x_user: Tensor) -> Tensor:
        x = F.relu(self.lin1(x_user))
        x = self.lin2(x)
        return x


class GAT_linear(torch.nn.Module):
    def __init__(self, hidden_channels, num_users, num_movies, num_heads=4):
        super().__init__()
        self.user_lin = torch.nn.Linear(5, hidden_channels)
        self.user_emb = torch.nn.Embedding(num_users, hidden_channels)
        self.movie_emb = torch.nn.Embedding(num_movies, hidden_channels)
        
        self.gat = GAT(hidden_channels, num_heads)
        self.classifier = Classifier()
        
        self.metadata = None
    
    def set_metadata(self, metadata):
        self.metadata = metadata
        self.gat = to_hetero(self.gat, metadata=metadata)
    
    def forward(self, data: HeteroData) -> Tensor:
        user_ids = data["user"].n_id
        movie_ids = data["movie"].n_id
        x_dict = {
            "user": self.user_emb(user_ids) + self.user_lin(data["user"].x),
            "movie": self.movie_emb(movie_ids),
        }        
        x_dict = self.gat(x_dict, data.edge_index_dict)
        
        return self.classifier(
            x_dict["user"],
            x_dict["movie"],
            data["user", "rates", "movie"].edge_label_index,
        )


class GAT_concat(torch.nn.Module):
    def __init__(self, hidden_channels, num_users, num_movies, 
                 user_feature_dim=5, num_heads=4):
        super().__init__()
        self.user_emb = torch.nn.Embedding(num_users, hidden_channels)
        self.movie_emb = torch.nn.Embedding(num_movies, hidden_channels)
        self.user_feat_lin = torch.nn.Linear(user_feature_dim, hidden_channels)
        self.user_combiner = torch.nn.Linear(hidden_channels * 2, hidden_channels)
        self.gat = GAT(hidden_channels, num_heads)
        self.classifier = Classifier()
        
        self.metadata = None
    
    def set_metadata(self, metadata):
        self.metadata = metadata
        self.gat = to_hetero(self.gat, metadata=metadata)
    
    def forward(self, data: HeteroData) -> Tensor:
        user_embedding = self.user_emb(data["user"].n_id)
        user_features = self.user_feat_lin(data["user"].x)
        user_features = F.relu(user_features)
        combined_user_vec = torch.cat([user_embedding, user_features], dim=1)
        initial_user_x = self.user_combiner(combined_user_vec)
        
        x_dict = {
            "user": initial_user_x,
            "movie": self.movie_emb(data["movie"].n_id),
        }
        x_dict = self.gat(x_dict, data.edge_index_dict)
        pred = self.classifier(
            x_dict["user"],
            x_dict["movie"],
            data["user", "rates", "movie"].edge_label_index,
        )
        return pred


class GAT_mtl(torch.nn.Module):
    def __init__(self, hidden_channels, num_users, num_movies, num_heads=4):
        super().__init__()
        
        self.user_lin = torch.nn.Linear(5, hidden_channels)
        self.user_emb = torch.nn.Embedding(num_users, hidden_channels)
        self.movie_emb = torch.nn.Embedding(num_movies, hidden_channels)
        self.gat = GAT(hidden_channels, num_heads)
        self.classifier = Classifier()
        self.personality_predictor = PersonalityPredictor(hidden_channels, out_channels=5)
        
        self.metadata = None
    
    def set_metadata(self, metadata):
        self.metadata = metadata
        self.gat = to_hetero(self.gat, metadata=metadata)
    
    def forward(self, data: HeteroData) -> dict:
        x_dict = {
            "user": self.user_emb(data["user"].n_id) + self.user_lin(data["user"].x),
            "movie": self.movie_emb(data["movie"].n_id),
        }
        final_x_dict = self.gat(x_dict, data.edge_index_dict)
        pred_link = self.classifier(
            final_x_dict["user"],
            final_x_dict["movie"],
            data["user", "rates", "movie"].edge_label_index,
        )       
        pred_personality = self.personality_predictor(final_x_dict["user"])
        
        return {"link": pred_link, "personality": pred_personality}


class GAT_attention_guided(torch.nn.Module):
    def __init__(self, hidden_channels, num_users, num_movies, num_heads=4):
        super().__init__()
        
        self.user_emb = torch.nn.Embedding(num_users, hidden_channels)
        self.movie_emb = torch.nn.Embedding(num_movies, hidden_channels)
        self.personality_transform = torch.nn.Linear(5, num_heads)   
        self.user_lin = torch.nn.Linear(5, hidden_channels)
        self.gat = GAT(hidden_channels, num_heads)  
        self.classifier = Classifier()
        
        self.metadata = None
    
    def set_metadata(self, metadata):
        self.metadata = metadata
        self.gat = to_hetero(self.gat, metadata=metadata)
    
    def forward(self, data: HeteroData) -> Tensor:
        personality_bias = self.personality_transform(data["user"].x)
        x_dict = {
            "user": self.user_emb(data["user"].n_id) + self.user_lin(data["user"].x),
            "movie": self.movie_emb(data["movie"].n_id),
        }
        x_dict = self.gat(x_dict, data.edge_index_dict)
        
        return self.classifier(
            x_dict["user"],
            x_dict["movie"],
            data["user", "rates", "movie"].edge_label_index,
        )