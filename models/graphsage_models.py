"""
GNN-based recommendation models
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import SAGEConv, to_hetero
from torch_geometric.data import HeteroData


class GNN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.conv1 = SAGEConv(hidden_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        
    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x


class Classifier(torch.nn.Module):
    def forward(self, x_user: Tensor, x_movie: Tensor, edge_label_index: Tensor) -> Tensor:
        edge_feat_user = x_user[edge_label_index[0]]
        edge_feat_movie = x_movie[edge_label_index[1]]
        return (edge_feat_user * edge_feat_movie).sum(dim=-1)
        # u = F.normalize(edge_feat_user, dim=-1)
        # v = F.normalize(edge_feat_movie, dim=-1)
        # return (u * v).sum(dim=-1)


class PersonalityPredictor(torch.nn.Module):
    def __init__(self, in_channels, out_channels=5):
        super().__init__()
        self.lin1 = torch.nn.Linear(in_channels, out_channels)
    
    def forward(self, x_user: Tensor) -> Tensor:
        x = F.relu(self.lin1(x_user))
        return x


class Model_graphsage(torch.nn.Module):
    def __init__(self, hidden_channels, num_users, num_movies):
        super().__init__()
        self.user_lin = torch.nn.Linear(5, hidden_channels)
        self.user_emb = torch.nn.Embedding(num_users, hidden_channels)
        self.movie_emb = torch.nn.Embedding(num_movies, hidden_channels)
        self.gnn = GNN(hidden_channels)
        self.classifier = Classifier()
        
        self.metadata = None
    
    def set_metadata(self, metadata):
        self.metadata = metadata
        self.gnn = to_hetero(self.gnn, metadata=metadata)
    
    def forward(self, data: HeteroData) -> Tensor:
        user_ids = data["user"].n_id
        movie_ids = data["movie"].n_id
        x_dict = {
            "user": self.user_emb(user_ids),
            "movie": self.movie_emb(movie_ids),
        }
        x_dict = self.gnn(x_dict, data.edge_index_dict)
        
        return self.classifier(
            x_dict["user"],
            x_dict["movie"],
            data["user", "rates", "movie"].edge_label_index,
        )


class Model_linear(torch.nn.Module):
    def __init__(self, hidden_channels, num_users, num_movies):
        super().__init__()
        self.user_lin = torch.nn.Linear(5, hidden_channels)
        self.user_emb = torch.nn.Embedding(num_users, hidden_channels)
        self.movie_emb = torch.nn.Embedding(num_movies, hidden_channels)
        self.gnn = GNN(hidden_channels)
        self.classifier = Classifier()
        
        self.metadata = None
    
    def set_metadata(self, metadata):
        self.metadata = metadata
        self.gnn = to_hetero(self.gnn, metadata=metadata)
    
    def forward(self, data: HeteroData) -> Tensor:
        user_ids = data["user"].n_id
        movie_ids = data["movie"].n_id
        
        x_dict = {
            "user": self.user_emb(user_ids) + self.user_lin(data["user"].x),
            "movie": self.movie_emb(movie_ids),
        }
        
        x_dict = self.gnn(x_dict, data.edge_index_dict)
        
        return self.classifier(
            x_dict["user"],
            x_dict["movie"],
            data["user", "rates", "movie"].edge_label_index,
        )


class Model_concat(torch.nn.Module):
    def __init__(self, hidden_channels, num_users, num_movies, user_feature_dim=5):
        super().__init__()
        self.user_emb = torch.nn.Embedding(num_users, hidden_channels)
        self.movie_emb = torch.nn.Embedding(num_movies, hidden_channels)
        self.user_feat_lin = torch.nn.Linear(user_feature_dim, hidden_channels)
        self.user_combiner = torch.nn.Linear(hidden_channels * 2, hidden_channels)
        self.gnn = GNN(hidden_channels)
        self.classifier = Classifier()
        
        self.metadata = None
    
    def set_metadata(self, metadata):
        self.metadata = metadata
        self.gnn = to_hetero(self.gnn, metadata=metadata)
    
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
        x_dict = self.gnn(x_dict, data.edge_index_dict)
        
        pred = self.classifier(
            x_dict["user"],
            x_dict["movie"],
            data["user", "rates", "movie"].edge_label_index,
        )
        return pred


class Model_mtl(torch.nn.Module):
    def __init__(self, hidden_channels, num_users, num_movies):
        super().__init__()
        
        self.user_lin = torch.nn.Linear(5, hidden_channels)
        self.user_emb = torch.nn.Embedding(num_users, hidden_channels)
        self.movie_emb = torch.nn.Embedding(num_movies, hidden_channels)
        self.gnn = GNN(hidden_channels)
        self.classifier = Classifier()
        self.personality_predictor = PersonalityPredictor(hidden_channels, out_channels=5)
        
        self.metadata = None
    
    def set_metadata(self, metadata):
        self.metadata = metadata
        self.gnn = to_hetero(self.gnn, metadata=metadata)
    
    def forward(self, data: HeteroData) -> dict:
        x_dict = {
            "user": self.user_emb(data["user"].n_id) + self.user_lin(data["user"].x),
            "movie": self.movie_emb(data["movie"].n_id),
        }
        final_x_dict = self.gnn(x_dict, data.edge_index_dict)
        pred_link = self.classifier(
            final_x_dict["user"],
            final_x_dict["movie"],
            data["user", "rates", "movie"].edge_label_index,
        )
        pred_personality = self.personality_predictor(final_x_dict["user"])
        
        return {"link": pred_link, "personality": pred_personality}