"""
Run Last.fm experiments with ablation
"""
import os
import sys
import json
import glob
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

from config.experiment_config import ExperimentConfig
from data.lastfm_dataset import LastFMDataset, split_edges, create_hetero_data
from models.lastfm_models import Model_without, Model_userfeat_add, Model_userfeat_concat, Model_userfeat_mtl
from training.lastfm_trainer import train_lastfm_model


def run_lastfm_experiment(config_path: str):
    """
    Run a single Last.fm experiment
    
    Args:
        config_path: Path to YAML config file
    
    Returns:
        Dictionary with results including ablation if applicable
    """
    # Load config
    config = ExperimentConfig.from_yaml(config_path)
    print("="*70)
    print(f"EXPERIMENT: {config.experiment_name}")
    print("="*70)
    print(f"Model: {config.model.name}")
    print(f"Features: {config.data.feature_type}")
    print("="*70)
    
    # Create save directory
    os.makedirs(config.save_dir, exist_ok=True)
    config.to_yaml(os.path.join(config.save_dir, 'config.yaml'))
    
    # Load data
    print("\n[1/5] Loading Last.fm data...")
    dataset = LastFMDataset(config)
    dataset.load_data()
    dataset.apply_sparsification(config.data.sparsity)
    dataset.create_mappings()
    dataset.process_user_features()
    
    # Get user features based on type
    print(f"\n[2/5] Preparing user features ({config.data.feature_type})...")
    if config.data.feature_type == 'none':
        user_features = None
        user_feat_dim = None
    else:
        user_features = dataset.get_user_features(config.data.feature_type)
        user_feat_dim = user_features.shape[1]
        print(f"User feature dimension: {user_feat_dim}")
    
    # Get edge index
    edge_index = dataset.get_edge_index()
    
    # Split edges
    print("\n[3/5] Splitting data...")
    train_edges, val_edges, test_edges = split_edges(
        edge_index,
        min_interactions=config.data.min_interactions,
        seed=config.training.seed
    )
    
    # Create HeteroData
    if user_features is None:
        # For Model_without, create dummy features
        user_features = torch.zeros(len(dataset.unique_user_id), 1)
    
    data = create_hetero_data(
        dataset.unique_user_id,
        dataset.unique_artist_id,
        user_features,
        train_edges
    )
    
    print(f"\nData summary:")
    print(f"  Users: {data['user'].num_nodes}")
    print(f"  Artists: {data['artist'].num_nodes}")
    print(f"  User features: {data['user'].x.shape[1]}")
    
    # Select model
    print(f"\n[4/5] Training {config.model.name}...")
    model_map = {
        'Model_without': Model_without,
        'Model_userfeat_add': Model_userfeat_add,
        'Model_userfeat_concat': Model_userfeat_concat,
        'Model_userfeat_mtl': Model_userfeat_mtl
    }
    
    model_class = model_map[config.model.name]
    
    # Run multiple trials
    all_results = []
    
    for run in range(config.training.num_runs):
        print(f"\n{'='*70}")
        print(f"Run {run+1}/{config.training.num_runs}")
        print('='*70)
        
        # Update seed for this run
        from copy import deepcopy
        run_config = deepcopy(config)
        run_config.training.seed = config.training.seed + run * 100
        
        # Train model
        results, model = train_lastfm_model(
            model_class=model_class,
            data=data,
            train_edges=train_edges,
            val_edges=val_edges,
            test_edges=test_edges,
            config=run_config,
            user_feat_dim=user_feat_dim,
            verbose=True
        )
        
        results['run'] = run + 1
        results['seed'] = run_config.training.seed
        all_results.append(results)
        
        print(f"\nRun {run+1} Results:")
        for k, v in results.items():
            if k not in ['run', 'seed']:
                print(f"  {k}: {v:.4f}")
    
    # Aggregate results
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
        'feature_type': config.data.feature_type,
        'num_runs': config.training.num_runs,
        'avg_metrics': avg_results,
        'std_metrics': std_results,
        'all_runs': all_results,
        'config': config.to_dict(),
        'timestamp': datetime.now().isoformat()
    }
    
    # Save results
    results_path = os.path.join(config.save_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print("\n" + "="*70)
    print(f"Average Results ({config.training.num_runs} runs):")
    for k, v in avg_results.items():
        std = std_results.get(f"{k}_std", 0)
        print(f"  {k}: {v:.4f} ± {std:.4f}")
    
    print(f"\n✓ Results saved to: {results_path}")
    print("="*70)
    
    return final_results


def run_all_lastfm_experiments(config_dir="config/experiments_lastfm/ablation", pattern="*.yaml"):
    """
    Run all Last.fm experiments and generate summary
    """
    # Find all config files
    config_pattern = os.path.join(config_dir, pattern)
    config_files = sorted(glob.glob(config_pattern))
    
    if not config_files:
        print(f"No config files found matching: {config_pattern}")
        sys.exit(1)
    
    print("="*70)
    print(f"LAST.FM EXPERIMENTS BATCH")
    print("="*70)
    print(f"Found {len(config_files)} config files")
    print(f"Each will run {3} times (different seeds)")
    print(f"Total training runs: {len(config_files) * 3}")
    print("="*70)
    
    # Run all experiments
    all_results = []
    failed_experiments = []
    
    for i, config_path in enumerate(config_files, 1):
        print(f"\n{'#'*70}")
        print(f"# EXPERIMENT {i}/{len(config_files)}: {os.path.basename(config_path)}")
        print(f"{'#'*70}\n")
        
        try:
            results = run_lastfm_experiment(config_path)
            all_results.append(results)
        except Exception as e:
            print(f"\n❌ ERROR in experiment {i}: {str(e)}")
            import traceback
            traceback.print_exc()
            failed_experiments.append(config_path)
            continue
    
    # Generate summary table
    print("\n" + "="*70)
    print("GENERATING SUMMARY TABLE")
    print("="*70)
    
    summary_data = []
    
    for res in all_results:
        row = {
            'experiment': res['experiment_name'],
            'model': res['model'],
            'feature_type': res['feature_type'],
            'num_runs': res['num_runs'],
        }
        
        # Add average metrics
        for k, v in res['avg_metrics'].items():
            row[k] = v
        
        # Add std metrics
        for k, v in res['std_metrics'].items():
            row[k] = v
        
        summary_data.append(row)
    
    summary_df = pd.DataFrame(summary_data)
    
    # Save summary
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = f"results/lastfm_ablation_summary_{timestamp}.csv"
    os.makedirs("results", exist_ok=True)
    summary_df.to_csv(summary_path, index=False)
    
    # Print formatted table
    print("\n" + "="*100)
    print("LAST.FM RESULTS")
    print("="*100)
    
    print(f"\n{'Model':<25} {'Features':<15} {'HR@10':<10} {'NDCG@10':<10}")
    print("-"*100)
    
    for _, row in summary_df.iterrows():
        print(f"{row['model']:<25} {row['feature_type']:<15} "
              f"{row['HR@10']:<10.4f} {row['NDCG@10']:<10.4f}")
    
    print("\n" + "="*100)
    print(f"\nSummary saved to: {summary_path}")
    
    # Print failures
    if failed_experiments:
        print(f"\n⚠️  {len(failed_experiments)} experiments failed:")
        for exp in failed_experiments:
            print(f"  - {os.path.basename(exp)}")
    
    print(f"\n✓ Completed {len(all_results)}/{len(config_files)} experiments successfully")
    print("="*70)
    
    return summary_df


if __name__ == "__main__":
    if len(sys.argv) > 1:
        config_dir = sys.argv[1]
        pattern = sys.argv[2] if len(sys.argv) > 2 else "*.yaml"
    else:
        config_dir = "config/experiments_lastfm/ablation"
        pattern = "*.yaml"
    
    print(f"Running Last.fm experiments from: {config_dir}/{pattern}\n")
    
    if os.path.isfile(config_dir):
        run_lastfm_experiment(config_dir)
    else:
        run_all_lastfm_experiments(config_dir, pattern)