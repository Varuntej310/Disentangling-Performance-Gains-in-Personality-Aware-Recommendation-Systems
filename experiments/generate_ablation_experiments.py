"""
Generate config files for ablation table experiments
Fills table with results across:
- Models: Linear, Concat, MTL
- Personality types: real, shuffled, uniform noise
- Sparsity levels: 5%, 30%, 60%, 100%
- With and without personality features (ablation)
"""
import os
import yaml
from pathlib import Path
from config.experiment_config import ExperimentConfig


def generate_ablation_table_configs(output_dir="config/experiments/ablation_table"):
    """
    Generate all config files needed for the ablation table
    
    Returns:
        List of generated config file paths
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Experiment parameters
    models = ['Model_linear', 'Model_concat', 'Model_mtl']
    personality_types = ['real', 'shuffled', 'noise']
    sparsity_levels = [5, 30, 60, 100]
    
    # Base configuration
    base_config = {
        'paths': {
            'ratings': "/home/fac/ashok.sairam/btp-varun/btp-1/ratings.csv",
            'personality': "/home/fac/ashok.sairam/btp-varun/btp-1/personality-data.csv"
        },
        'data': {
            'name': 'movielens',
            'min_interactions': 2,
            'personality_columns': ['openness', 'agreeableness', 'emotional_stability',
                                   'conscientiousness', 'extraversion']
        },
        'model': {
            'hidden_channels': 5,
            'num_neighbors': [20, 10],
            'personality_loss_weight': 0.5  # for MTL
        },
        'training': {
            'num_epochs': 50,
            'batch_size': 512,
            'lr': 0.001,
            'patience': 12,
            'num_runs': 3,
            'seed': 42,
            'device': 'cuda',
            'neg_sampling_ratio': 2.0
        },
        'eval': {
            'k_list': [3, 5, 10],
            'num_negatives': 99
        }
    }
    
    generated_files = []
    
    # Generate configs for each combination
    for model in models:
        for pers_type in personality_types:
            for sparsity in sparsity_levels:
                # Create config for this combination
                config = base_config.copy()
                
                # Set model
                config['model'] = base_config['model'].copy()
                config['model']['name'] = model
                
                # Set data parameters
                config['data'] = base_config['data'].copy()
                config['data']['sparsity_percentile'] = float(sparsity)
                config['data']['personality_type'] = pers_type
                
                # For noise, set distribution
                if pers_type == 'noise':
                    config['data']['noise_distribution'] = 'uniform'
                else:
                    config['data']['noise_distribution'] = None
                
                # Create experiment name
                model_short = model.replace('Model_', '').lower()
                pers_short = {
                    'real': 'real',
                    'shuffled': 'shuf',
                    'noise': 'noise'
                }[pers_type]
                
                exp_name = f"{model_short}_{pers_short}_sp{sparsity}"
                config['experiment_name'] = exp_name
                config['save_dir'] = f"results/ablation_table/{exp_name}"
                
                # Copy other configs
                config['training'] = base_config['training'].copy()
                config['eval'] = base_config['eval'].copy()
                config['paths'] = base_config['paths'].copy()
                
                # Save config
                filename = f"{exp_name}.yaml"
                filepath = os.path.join(output_dir, filename)
                
                with open(filepath, 'w') as f:
                    yaml.dump(config, f, default_flow_style=False, sort_keys=False)
                
                generated_files.append(filepath)
                print(f"Generated: {filename}")

    # for Model_graphsage
    for sparsity in sparsity_levels:
        # Create config for this combination
        config = base_config.copy()
        
        # Set model
        config['model'] = base_config['model'].copy()
        config['model']['name'] = 'Model_graphsage'
        
        # Set data parameters
        config['data'] = base_config['data'].copy()
        config['data']['sparsity_percentile'] = float(sparsity)
        config['data']['personality_type'] = 'none'
        config['data']['noise_distribution'] = None
        
        # Create experiment name
        model_short = 'graphsage'
        
        exp_name = f"{model_short}_sp{sparsity}"
        config['experiment_name'] = exp_name
        config['save_dir'] = f"results/ablation_table/{exp_name}"
        
        # Copy other configs
        config['training'] = base_config['training'].copy()
        config['eval'] = base_config['eval'].copy()
        config['paths'] = base_config['paths'].copy()
        
        # Save config
        filename = f"{exp_name}.yaml"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        generated_files.append(filepath)
        print(f"Generated: {filename}")

    print(f"\n✓ Generated {len(generated_files)} config files in {output_dir}")
    return generated_files


def create_ablation_analysis_script(output_file="analyze_ablation_results.py"):
    """
    Create a script to analyze results and extract ablation metrics
    """
    script_content = '''"""
Analyze ablation experiment results and compute personality vs zeroed metrics
"""
import json
import pandas as pd
import numpy as np
from pathlib import Path
import torch
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from config.experiment_config import ExperimentConfig
from data.base_dataset import MovieLensDataset, split_edges, create_hetero_data
from experiments.ablations import evaluate_personality_ablation


def load_trained_model(result_dir, model_class):
    """Load a trained model from results directory"""
    # This is a placeholder - you may need to save/load model checkpoints
    # For now, we'll retrain or use the existing results
    pass


def compute_ablation_for_experiment(config_path):
    """
    Compute ablation metrics (real vs zeroed personality) for a single experiment
    
    Returns:
        Dict with both real and zeroed personality metrics
    """
    print(f"\\nProcessing: {config_path}")
    
    # Load config
    config = ExperimentConfig.from_yaml(config_path)
    
    # Load existing results
    results_path = Path(config.save_dir) / 'results.json'
    
    if not results_path.exists():
        print(f"  Warning: Results not found at {results_path}")
        return None
    
    with open(results_path) as f:
        results = json.load(f)
    
    # Check if ablation was already computed
    if 'ablation' in results:
        print(f"  ✓ Ablation already computed")
        return results
    
    # Otherwise, need to load model and compute
    # For now, we'll use the regular results and need to add ablation computation
    print(f"  ⚠ Ablation needs to be computed - run with --compute-ablation flag")
    
    return results


def extract_metrics_for_table(all_results):
    """
    Extract and organize metrics for the ablation table
    
    Returns:
        DataFrame with columns: model, personality_type, sparsity, metric, value, value_zeroed
    """
    rows = []
    
    for exp_name, result_data in all_results.items():
        if result_data is None:
            continue
        
        # Parse experiment name
        parts = exp_name.replace('.yaml', '').split('_')
        model = parts[0]  # linear, concat, mtl
        pers_type = parts[1]  # real, shuf, noise
        sparsity = parts[2].replace('sp', '')  # 1, 30, 60, 100
        
        # Get metrics
        avg_metrics = result_data.get('avg_metrics', {})
        
        # If ablation exists, get zeroed metrics
        ablation = result_data.get('ablation', {})
        zeroed_metrics = ablation.get('zero', {}) if ablation else {}
        
        # Add rows for each metric
        for metric in ['HR@5', 'NDCG@5', 'HR@10', 'NDCG@10']:
            rows.append({
                'model': model,
                'personality_type': pers_type,
                'sparsity': int(sparsity),
                'metric': metric,
                'value': avg_metrics.get(metric, np.nan),
                'value_zeroed': zeroed_metrics.get(metric, np.nan)
            })
    
    return pd.DataFrame(rows)


def format_latex_table(df):
    """
    Format results as LaTeX table matching the paper format
    """
    # Pivot to get the structure we need
    table_rows = []
    
    models = [
        ('linear', 'real', 'Linear + org. personality'),
        ('linear', 'shuf', 'Linear + shuffled personality'),
        ('linear', 'noise', 'Linear + uniform noise'),
        ('concat', 'real', 'Concatenation + org. personality'),
        ('concat', 'shuf', 'Concatenation + shuffled personality'),
        ('concat', 'noise', 'Concatenation + uniform noise'),
        ('mtl', 'real', 'MTL + org. personality'),
        ('mtl', 'shuf', 'MTL + shuffled personality'),
        ('mtl', 'noise', 'MTL + uniform noise'),
    ]
    
    print("\\n" + "="*100)
    print("ABLATION TABLE RESULTS")
    print("="*100)
    
    for model_key, pers_key, model_label in models:
        print(f"\\n{model_label}")
        print("-" * 100)
        
        for metric in ['HR@5', 'NDCG@5', 'HR@10', 'NDCG@10']:
            subset = df[(df['model'] == model_key) & 
                       (df['personality_type'] == pers_key) & 
                       (df['metric'] == metric)]
            
            if len(subset) == 0:
                continue
            
            # Sort by sparsity
            subset = subset.sort_values('sparsity')
            
            # Build row
            row_parts = [f"  {metric:8s}"]
            
            # Real personality values
            for sp in [1, 30, 60, 100]:
                val = subset[subset['sparsity'] == sp]['value'].values
                if len(val) > 0 and not np.isnan(val[0]):
                    row_parts.append(f"{val[0]:.4f}")
                else:
                    row_parts.append("0.XXX")
            
            # Zeroed personality values
            for sp in [1, 30, 60, 100]:
                val = subset[subset['sparsity'] == sp]['value_zeroed'].values
                if len(val) > 0 and not np.isnan(val[0]):
                    row_parts.append(f"{val[0]:.4f}")
                else:
                    row_parts.append("0.XXX")
            
            print("  ".join(row_parts))
    
    print("\\n" + "="*100)
    
    # Save to CSV
    output_path = 'results/ablation_table_results.csv'
    df.to_csv(output_path, index=False)
    print(f"\\nResults saved to: {output_path}")
    
    return df


def main():
    """Main analysis function"""
    import glob
    
    # Find all config files
    config_dir = "config/experiments/ablation_table"
    config_files = sorted(glob.glob(f"{config_dir}/*.yaml"))
    
    print(f"Found {len(config_files)} experiment configs")
    
    # Load results for each experiment
    all_results = {}
    
    for config_path in config_files:
        exp_name = Path(config_path).stem
        result = compute_ablation_for_experiment(config_path)
        all_results[exp_name] = result
    
    # Extract metrics
    df = extract_metrics_for_table(all_results)
    
    # Format and display
    format_latex_table(df)
    
    # Summary statistics
    print(f"\\nSummary:")
    print(f"  Total experiments: {len(all_results)}")
    print(f"  Completed: {sum(1 for r in all_results.values() if r is not None)}")
    print(f"  Missing: {sum(1 for r in all_results.values() if r is None)}")


if __name__ == "__main__":
    main()
'''
    
    with open(output_file, 'w') as f:
        f.write(script_content)
    
    print(f"\n✓ Created analysis script: {output_file}")


def print_run_instructions():
    """Print instructions for running the experiments"""
    print("\n" + "="*70)
    print("INSTRUCTIONS TO RUN ABLATION TABLE EXPERIMENTS")
    print("="*70)
    
    print("\n1. Run all experiments:")
    print("   python run_batch_experiments.py config/experiments/ablation_table/")
    
    print("\n2. This will run 36 experiments total:")
    print("   - 3 models (Linear, Concat, MTL)")
    print("   - 3 personality types (real, shuffled, noise)")
    print("   - 4 sparsity levels (1%, 30%, 60%, 100%)")
    print("   = 3 × 3 × 4 = 36 experiments")
    
    print("\n3. Each experiment runs 3 times (different seeds)")
    print("   Total training runs: 36 × 3 = 108")
    
    print("\n4. Analyze results:")
    print("   python analyze_ablation_results.py")
    
    print("\n5. Results will be saved to:")
    print("   - Individual: results/ablation_table/<exp_name>/")
    print("   - Summary: results/ablation_table_results.csv")
    print("   - Batch summary: results/batch_summary_*.csv")
    
    print("\n" + "="*70)
    print("ESTIMATED TIME")
    print("="*70)
    print("If each experiment takes ~10 minutes:")
    print("Total time: 36 × 10 = 360 minutes ≈ 6 hours")
    print("\nRecommendation: Run overnight or on multiple GPUs")
    print("="*70)


if __name__ == "__main__":
    print("="*70)
    print("GENERATING ABLATION TABLE EXPERIMENT CONFIGS")
    print("="*70)
    
    # Generate configs
    generated_files = generate_ablation_table_configs()
    
    # Create analysis script
    create_ablation_analysis_script()
    
    # Print instructions
    print_run_instructions()
    
    print("\n✓ Setup complete! Ready to run experiments.")