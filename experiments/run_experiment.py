"""
Single experiment runner from YAML config
"""
import os
import sys
import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime

from config.experiment_config import ExperimentConfig
from data.base_dataset import MovieLensDataset, split_edges, create_hetero_data
from models.graphsage_models import Model_linear, Model_concat, Model_mtl, Model_graphsage
from models.lightgcn import LightGCN
from training.trainer import train_gnn_model, train_lightgcn


def run_single_experiment(config_path: str):
    config = ExperimentConfig.from_yaml(config_path)
    print("="*70)
    print("STARTING EXPERIMENT")
    print("="*70)
    print(config)
    print("="*70)
    
    os.makedirs(config.save_dir, exist_ok=True)
    config.to_yaml(os.path.join(config.save_dir, 'config.yaml'))
    
    print("\n[1/5] Loading data...")
    dataset = MovieLensDataset(config)
    dataset.load_data()
    dataset.sparsify_ratings()
    dataset.create_mappings()
    dataset.merge_features()
    
    edge_index = dataset.get_edge_index()
    print("\n[2/5] Splitting data...")
    train_edges, val_edges, test_edges = split_edges(
        edge_index,
        min_interactions=config.data.min_interactions,
        seed=config.training.seed
    )

    data = create_hetero_data(
        dataset.unique_user_id,
        dataset.unique_movie_id,
        dataset.filtered_personality,
        train_edges
    )
    
    print(f"\nData summary:")
    print(f"  Users: {data['user'].num_nodes}")
    print(f"  Movies: {data['movie'].num_nodes}")
    print(f"  Personality features: {data['user'].x.shape[1]}")

    print(f"\n[3/5] Initializing model: {config.model.name}")
    model_map = {
        'Model_graphsage': Model_graphsage,
        'Model_linear': Model_linear,
        'Model_concat': Model_concat,
        'Model_mtl': Model_mtl,
        'LightGCN': LightGCN,
    }
    
    if config.model.name not in model_map:
        raise ValueError(f"Unknown model: {config.model.name}")
    
    model_class = model_map[config.model.name]
    
    print(f"\n[4/5] Training ({config.training.num_runs} runs)...")
    all_results = []
    
    for run in range(config.training.num_runs):
        print(f"\n{'='*70}")
        print(f"Run {run+1}/{config.training.num_runs}")
        print('='*70)
        
        run_seed = config.training.seed + run * 100
        
        from copy import deepcopy
        run_config = deepcopy(config)
        run_config.training.seed = run_seed
        
        if config.model.name == 'LightGCN':
            results, model = train_lightgcn(
                data=data,
                train_edges=train_edges,
                val_edges=val_edges,
                test_edges=test_edges,
                config=run_config,
                verbose=True
            )
        else:
            results, model = train_gnn_model(
                model_class=model_class,
                data=data,
                train_edges=train_edges,
                val_edges=val_edges,
                test_edges=test_edges,
                config=run_config,
                personality_loss_weight=config.model.personality_loss_weight,
                verbose=True
            )
        
        results['run'] = run + 1
        results['seed'] = run_seed
        all_results.append(results)
        
        print(f"\nRun {run+1} Results:")
        for k, v in results.items():
            if k not in ['run', 'seed']:
                print(f"  {k}: {v:.4f}")
    
    print(f"\n[5/5] Aggregating results...")
    avg_results = {}
    std_results = {}
    
    for key in all_results[0].keys():
        if key not in ['run', 'seed']:
            values = [r[key] for r in all_results]
            avg_results[key] = np.mean(values)
            std_results[f"{key}_std"] = np.std(values)
    
    final_results = {
        'experiment_name': config.experiment_name,
        'model': config.model.name,
        'num_runs': config.training.num_runs,
        'avg_metrics': avg_results,
        'std_metrics': std_results,
        'all_runs': all_results,
        'config': config.to_dict(),
        'timestamp': datetime.now().isoformat()
    }
    
    results_path = os.path.join(config.save_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print("\n" + "="*70)
    print("EXPERIMENT COMPLETED")
    print("="*70)
    print(f"\nAverage Results ({config.training.num_runs} runs):")
    for k, v in avg_results.items():
        std = std_results.get(f"{k}_std", 0)
        print(f"  {k}: {v:.4f} ± {std:.4f}")
    
    print(f"\nResults saved to: {results_path}")
    print("="*70)
    
    return final_results


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python run_experiment.py <config_path>")
        print("Example: python run_experiment.py config/experiments/baseline.yaml")
        sys.exit(1)
    
    config_path = sys.argv[1]
    
    if not os.path.exists(config_path):
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)
    
    run_single_experiment(config_path)