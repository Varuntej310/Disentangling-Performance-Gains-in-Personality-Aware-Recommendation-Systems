"""
Main script to run experiments
"""
import torch
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Import configuration
from config.experiment_config import Config

# Import data utilities
from data.base_dataset import (
    MovieLensDataset, 
    split_edges, 
    create_hetero_data
)

# Import models
from models.graphsage_models import Model_linear, Model_concat, Model_mtl

# Import training utilities
from training.trainer import train_gnn_model, train_lightgcn

# Import experiment utilities
from experiments.run_experiment import (
    run_multiple_trials,
    sweep_lambda,
    sweep_noise_distributions,
    compare_models
)

# Import ablation utilities
from experiments.ablations import evaluate_personality_ablation


def main():
    config = Config()
    
    print("="*60)
    print("Loading and preprocessing data...")
    print("="*60)
    
    # Load and preprocess 
    dataset = MovieLensDataset(config)
    dataset.load_data()
    dataset.sparsify_ratings()
    dataset.create_mappings()
    dataset.merge_features()
    
    # Get edge index
    edge_index = dataset.get_edge_index()
    
    # Split edges
    train_edges, val_edges, test_edges = split_edges(
        edge_index, 
        min_interactions=config.data.min_interactions,
        seed=config.training.seed
    )
    
    # Create HeteroData
    data = create_hetero_data(
        dataset.unique_user_id,
        dataset.unique_movie_id,
        dataset.filtered_personality,
        train_edges
    )
    
    print(f"\nData summary:")
    print(f"  Users: {data['user'].num_nodes}")
    print(f"  Movies: {data['movie'].num_nodes}")
    print(f"  Train edges: {len(train_edges)}")
    print(f"  Val edges: {len(val_edges)}")
    print(f"  Test edges: {len(test_edges)}")
    print(f"  Personality features: {data['user'].x.shape[1]}")
    
    
    
    # Experiment 1: Compare model architectures
    
    # print("\n" + "="*60)
    # print("Experiment 1: Comparing model architectures")
    # print("="*60)
    
    # model_comparison_df = compare_models(
    #     model_classes=[Model_linear, Model_concat, Model_mtl],
    #     data=data,
    #     train_edges=train_edges,
    #     val_edges=val_edges,
    #     test_edges=test_edges,
    #     config=config,
    #     num_runs=3,
    #     verbose=True
    # )
    
    # print("\nModel Comparison Results:")
    # print(model_comparison_df[['model', 'HR@10', 'NDCG@10']])
    # model_comparison_df.to_csv('results/model_comparison.csv', index=False)
    
    
    
    # Experiment 2: LightGCN baseline

    print("\n" + "="*60)
    print("Experiment 2: Training LightGCN baseline")
    print("="*60)
    
    lightgcn_results, _ = train_lightgcn(
        data=data,
        train_edges=train_edges,
        val_edges=val_edges,
        test_edges=test_edges,
        config=config,
        verbose=True
    )
    
    print("\nLightGCN Results:")
    for k, v in lightgcn_results.items():
        print(f"  {k}: {v:.4f}")
    
    
    
    # Experiment 3: Lambda sweep for MTL

    print("\n" + "="*60)
    print("Experiment 3: Sweeping personality loss weight (lambda)")
    print("="*60)
    
    lambda_sweep_df = sweep_lambda(
        model_class=Model_mtl,
        data=data,
        train_edges=train_edges,
        val_edges=val_edges,
        test_edges=test_edges,
        config=config,
        lambdas=[0.0, 0.25, 0.5, 0.75, 1.0],
        num_runs=3,
        verbose=True
    )
    
    print("\nLambda Sweep Results:")
    print(lambda_sweep_df[['lambda', 'HR@10', 'NDCG@10']])
    lambda_sweep_df.to_csv('results/lambda_sweep.csv', index=False)
    
    
    
    # Experiment 4: Personality ablation
    
    print("\n" + "="*60)
    print("Experiment 4: Personality feature ablation")
    print("="*60)
    
    # Train best model
    best_results, best_model = train_gnn_model(
        Model_linear,
        data,
        train_edges,
        val_edges,
        test_edges,
        config,
        verbose=True
    )
    
    # Run ablation
    device = torch.device(config.DEVICE if torch.cuda.is_available() else 'cpu')
    ablation_results = evaluate_personality_ablation(
        best_model,
        Model_linear,
        data.to(device),
        test_edges,
        np.vstack([train_edges, val_edges]),
        config
    )
    
    print("\nPersonality Ablation Results:")
    print(f"  With real personality: NDCG@10 = {ablation_results['real']['NDCG@10']:.4f}")
    print(f"  With zero personality: NDCG@10 = {ablation_results['zero']['NDCG@10']:.4f}")
    print(f"  Delta: {ablation_results['delta']['NDCG@10']:.4f}")
    
    
    
    # Experiment 5: Noise distribution sweep
    
    print("\n" + "="*60)
    print("Experiment 5: Testing different noise distributions")
    print("="*60)
    
    noise_sweep_df = sweep_noise_distributions(
        model_class=Model_mtl,
        data=data,
        train_edges=train_edges,
        val_edges=val_edges,
        test_edges=test_edges,
        config=config,
        distributions=["uniform", "normal", "laplace", "bernoulli", "exponential"],
        num_runs=2,
        personality_loss_weight=0.25,
        verbose=True
    )
    
    print("\nNoise Distribution Results:")
    print(noise_sweep_df[['distribution', 'HR@10', 'NDCG@10']])
    noise_sweep_df.to_csv('results/noise_distributions.csv', index=False)
    
    
    print("\n" + "="*60)
    print("All experiments completed!")
    print("="*60)


if __name__ == "__main__":
    main()