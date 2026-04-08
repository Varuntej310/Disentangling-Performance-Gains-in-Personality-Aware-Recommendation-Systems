"""
YAML-based experiment configuration
"""
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import yaml
import os


@dataclass
class PathConfig:
    """Data paths"""
    ratings: str = "/home/fac/ashok.sairam/btp-varun/btp-1/ratings.csv"
    personality: str = "/home/fac/ashok.sairam/btp-varun/btp-1/personality-data.csv"
    interactions: str = "/home/fac/ashok.sairam/btp-varun/btp-1/lastfm_9000_users.csv"
    user_features: str = "/home/fac/ashok.sairam/btp-varun/btp-1/user_features_9000.csv"

@dataclass
class DataConfig:
    """Data processing configuration"""
    name: str = "movielens"
    sparsity_percentile: float = 1.0
    sparsity: float = 100.0
    dataset_type: str = "movielens"  # movielens or lastfm
    feature_type: str = "real"  # real, random, shuffled, zero
    min_interactions: int = 2
    personality_type: str = "real"  # real, shuffled, random noise
    noise_distribution: Optional[str] = None  # uniform, normal, laplace, bernoulli, exponential
    personality_columns: List[str] = field(default_factory=lambda: [
        'openness', 'agreeableness', 'emotional_stability',
        'conscientiousness', 'extraversion'
    ])
    
    
@dataclass
class ModelConfig:
    """Model architecture configuration"""
    name: str = "Model_linear"  # Model_linear, Model_concat, Model_mtl, LightGCN
    hidden_channels: int = 5
    num_neighbors: List[int] = field(default_factory=lambda: [20, 10])
    
    # MTL specific
    personality_loss_weight: float = 0.0
    
    # LightGCN specific
    num_layers: int = 3
    embedding_dim: int = 64


@dataclass
class TrainingConfig:
    """Training configuration"""
    num_epochs: int = 50
    batch_size: int = 128
    lr: float = 1e-3
    patience: int = 7
    num_runs: int = 1
    seed: int = 42
    device: str = "cuda"
    neg_sampling_ratio: float = 2.0


@dataclass
class EvalConfig:
    """Evaluation configuration"""
    k_list: List[int] = field(default_factory=lambda: [3, 5, 10])
    num_negatives: int = 99


@dataclass
class ExperimentConfig:
    """Complete experiment configuration"""
    experiment_name: str = "baseline"
    paths: PathConfig = field(default_factory=PathConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    save_dir: str = "results"
    
    @classmethod
    def from_yaml(cls, path: str):
        """Load configuration from YAML file"""
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]):
        """Create config from dictionary"""
        paths = PathConfig(**config_dict.get('paths', {}))
        data = DataConfig(**config_dict.get('data', {}))
        model = ModelConfig(**config_dict.get('model', {}))
        training = TrainingConfig(**config_dict.get('training', {}))
        eval_cfg = EvalConfig(**config_dict.get('eval', {}))
        
        return cls(
            experiment_name=config_dict.get('experiment_name', 'baseline'),
            paths=paths,
            data=data,
            model=model,
            training=training,
            eval=eval_cfg,
            save_dir=config_dict.get('save_dir', 'results')
        )
    
    def to_dict(self):
        """Convert config to dictionary"""
        return {
            'experiment_name': self.experiment_name,
            'paths': self.paths.__dict__,
            'data': self.data.__dict__,
            'model': self.model.__dict__,
            'training': self.training.__dict__,
            'eval': self.eval.__dict__,
            'save_dir': self.save_dir
        }
    
    def to_yaml(self, path: str):
        """Save configuration to YAML file"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
    
    def __repr__(self):
        """Pretty print configuration"""
        lines = [
            f"Experiment: {self.experiment_name}",
            f"  Model: {self.model.name}",
            f"  Hidden: {self.model.hidden_channels}",
            f"  Epochs: {self.training.num_epochs}",
            f"  LR: {self.training.lr}",
            f"  Personality: {self.data.personality_type}",
            f"  Lambda: {self.model.personality_loss_weight}",
        ]
        return "\n".join(lines)