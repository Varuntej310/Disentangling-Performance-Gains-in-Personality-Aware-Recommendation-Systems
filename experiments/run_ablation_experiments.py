"""
Run ablation experiments with automatic computation of personality vs zeroed metrics
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
from data.base_dataset import MovieLensDataset, split_edges, create_hetero_data
from models.graphsage_models import Model_linear, Model_concat, Model_mtl
from training.trainer import train_gnn_model
from experiments.ablations import evaluate_personality_ablation


def run_experiment_with_ablation(config_path: str):
    config = ExperimentConfig.from_yaml(config_path)
    print("="*70)
    print(f"EXPERIMENT: {config.experiment_name}")
    print("="*70)
    print(config)
    print("="*70)
    os.makedirs(config.save_dir, exist_ok=True)
    config.to_yaml(os.path.join(config.save_dir, 'config.yaml'))

    print("\n[1/6] Loading data...")
    dataset = MovieLensDataset(config)
    dataset.load_data()
    dataset.sparsify_ratings()
    dataset.create_mappings()
    dataset.merge_features()
    
    edge_index = dataset.get_edge_index()

    print("\n[2/6] Splitting data...")
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
    print(f"  Sparsity: {config.data.sparsity_percentile}%")
    print(f"  Personality: {config.data.personality_type}")
    
    print(f"\n[3/6] Initializing model: {config.model.name}")
    model_map = {
        'Model_linear': Model_linear,
        'Model_concat': Model_concat,
        'Model_mtl': Model_mtl,
    }
    
    model_class = model_map[config.model.name]
    
    print(f"\n[4/6] Training ({config.training.num_runs} runs)...")
    all_results = []
    all_models = []
    
    for run in range(config.training.num_runs):
        print(f"\n{'='*70}")
        print(f"Run {run+1}/{config.training.num_runs}")
        print('='*70)
        
        from copy import deepcopy
        run_config = deepcopy(config)
        run_config.training.seed = config.training.seed + run * 100
        
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
        results['seed'] = run_config.training.seed
        all_results.append(results)
        all_models.append(model)
        
        print(f"\nRun {run+1} Results:")
        for k, v in results.items():
            if k not in ['run', 'seed']:
                print(f"  {k}: {v:.4f}")
    
    print(f"\n[5/6] Aggregating results...")
    avg_results = {}
    std_results = {}
    
    for key in all_results[0].keys():
        if key not in ['run', 'seed']:
            values = [r[key] for r in all_results]
            avg_results[key] = np.mean(values)
            std_results[f"{key}_std"] = np.std(values)
    
    print(f"\nAverage Results ({config.training.num_runs} runs):")
    for k, v in avg_results.items():
        std = std_results.get(f"{k}_std", 0)
        print(f"  {k}: {v:.4f} ± {std:.4f}")
    
    # Compute ablation metrics (personality vs zeroed)
    print(f"\n[6/6] Computing ablation (personality vs zeroed)...")
    best_model = all_models[-1]
    device = torch.device(config.training.device if torch.cuda.is_available() else 'cpu')
    
    ablation_train = np.vstack([train_edges, val_edges])
    
    ablation_results = evaluate_personality_ablation(
        best_model,
        model_class,
        data.to(device),
        test_edges,
        ablation_train,
        config
    )
    
    print("\nAblation Results:")
    print(f"  With real personality:")
    for k, v in ablation_results['real'].items():
        print(f"    {k}: {v:.4f}")
    
    print(f"  With zeroed personality:")
    for k, v in ablation_results['zero'].items():
        print(f"    {k}: {v:.4f}")
    
    print(f"  Delta (real - zero):")
    for k, v in ablation_results['delta'].items():
        print(f"    {k}: {v:+.4f}")

    final_results = {
        'experiment_name': config.experiment_name,
        'model': config.model.name,
        'personality_type': config.data.personality_type,
        'sparsity': config.data.sparsity_percentile,
        'num_runs': config.training.num_runs,
        'avg_metrics': avg_results,
        'std_metrics': std_results,
        'ablation': ablation_results,
        'all_runs': all_results,
        'config': config.to_dict(),
        'timestamp': datetime.now().isoformat()
    }
    
    results_path = os.path.join(config.save_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\n✓ Results saved to: {results_path}")
    
    return final_results


def run_all_ablation_experiments(config_dir="config/experiments/ablation_table", pattern="*.yaml"):
    config_pattern = os.path.join(config_dir, pattern)
    config_files = sorted(glob.glob(config_pattern))
    
    if not config_files:
        print(f"No config files found matching: {config_pattern}")
        sys.exit(1)
    
    print("="*70)
    print(f"ABLATION TABLE EXPERIMENT BATCH")
    print("="*70)
    print(f"Found {len(config_files)} config files")
    print(f"Each will run {3} times (different seeds)")
    print(f"Total training runs: {len(config_files) * 3}")
    print("="*70)
    
    all_results = []
    failed_experiments = []
    
    for i, config_path in enumerate(config_files, 1):
        print(f"\n{'#'*70}")
        print(f"# EXPERIMENT {i}/{len(config_files)}: {os.path.basename(config_path)}")
        print(f"{'#'*70}\n")
        
        try:
            results = run_experiment_with_ablation(config_path)
            all_results.append(results)
        except Exception as e:
            print(f"\n❌ ERROR in experiment {i}: {str(e)}")
            import traceback
            traceback.print_exc()
            failed_experiments.append(config_path)
            continue
    
    print("\n" + "="*70)
    print("GENERATING SUMMARY TABLE")
    print("="*70)
    
    summary_data = []
    
    for res in all_results:
        row = {
            'experiment': res['experiment_name'],
            'model': res['model'],
            'personality': res['personality_type'],
            'sparsity': res['sparsity'],
            'num_runs': res['num_runs'],
        }
        
        for k, v in res['avg_metrics'].items():
            row[k] = v
        
        if 'ablation' in res and 'zero' in res['ablation']:
            for k, v in res['ablation']['zero'].items():
                row[f"{k}_zeroed"] = v
        
        if 'ablation' in res and 'delta' in res['ablation']:
            for k, v in res['ablation']['delta'].items():
                row[f"{k}_delta"] = v
        
        summary_data.append(row)
    
    summary_df = pd.DataFrame(summary_data)
    
    summary_df = summary_df.sort_values(['model', 'personality', 'sparsity'])
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = f"results/ablation_table_summary_{timestamp}.csv"
    os.makedirs("results", exist_ok=True)
    summary_df.to_csv(summary_path, index=False)
    
    print("\n" + "="*100)
    print("ABLATION TABLE RESULTS")
    print("="*100)
    
    # Group by model and personality type
    for model in ['Model_linear', 'Model_concat', 'Model_mtl']:
        model_short = model.replace('Model_', '')
        
        for pers in ['real', 'shuffled', 'noise']:
            subset = summary_df[
                (summary_df['model'] == model) & 
                (summary_df['personality'] == pers)
            ]
            
            if len(subset) == 0:
                continue
            
            pers_label = {
                'real': 'org. personality',
                'shuffled': 'shuffled personality',
                'noise': 'uniform noise'
            }[pers]
            
            print(f"\n{model_short} + {pers_label}")
            print("-" * 100)
            
            for metric in ['HR@5', 'NDCG@5', 'HR@10', 'NDCG@10']:
                parts = [f"  {metric:8s}"]
                
                # Values with personality
                for sp in [1, 30, 60, 100]:
                    row = subset[subset['sparsity'] == sp]
                    if len(row) > 0:
                        val = row[metric].values[0]
                        parts.append(f"{val:.4f}")
                    else:
                        parts.append("  ---  ")
                
                # Values with zeroed personality
                for sp in [1, 30, 60, 100]:
                    row = subset[subset['sparsity'] == sp]
                    if len(row) > 0:
                        val = row[f"{metric}_zeroed"].values[0]
                        parts.append(f"{val:.4f}")
                    else:
                        parts.append("  ---  ")
                
                print("  ".join(parts))
    
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
        config_dir = "config/experiments/ablation_table"
        pattern = "*.yaml"
    
    print(f"Running ablation experiments from: {config_dir}/{pattern}\n")
    
    run_all_ablation_experiments(config_dir, pattern)