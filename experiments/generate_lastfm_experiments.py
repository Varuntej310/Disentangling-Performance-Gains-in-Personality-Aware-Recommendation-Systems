"""
Generate experiment configs for Last.fm dataset
"""
import os
import yaml
from pathlib import Path


def create_base_lastfm_config():
    """Base configuration for Last.fm experiments"""
    return {
        'paths': {
            'interactions': "/home/fac/ashok.sairam/btp-varun/btp-1/lastfm_9000_users.csv",
            'user_features': "/home/fac/ashok.sairam/btp-varun/btp-1/user_features_9000.csv",
        },
        'data': {
            'name': 'lastfm',
            'min_interactions': 2,
            'dataset_type': 'lastfm'  # To distinguish from movielens
        },
        'model': {
            'hidden_channels': 64,
            'num_neighbors': [20, 10],
        },
        'training': {
            'num_epochs': 30,
            'batch_size': 512,
            'lr': 0.001,
            'patience': 5,
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


def generate_lastfm_ablation_configs(output_dir="config/experiments_lastfm/ablation"):
    """
    Generate Last.fm ablation experiment configs
    
    Models: Model_without, Model_userfeat_add, Model_userfeat_concat
    Features: real, random (ablation is: real vs random)
    
    Total: 1 (without) + 2 (add: real, random) + 2 (concat: real, random) = 5 experiments
    """
    os.makedirs(output_dir, exist_ok=True)
    
    generated_files = []
    
    print("="*70)
    print("Generating Last.fm Ablation Experiments with Sparsification")
    print("="*70)
    
    sparsities = [0.1, 0.5, 2.5, 100]
    feature_types = ['real', 'random', 'shuffled']
    
    # 1. Model without features (baseline) - for all sparsities
    for sparsity in sparsities:
        config = create_base_lastfm_config()
        config['model']['name'] = 'Model_without'
        config['data']['feature_type'] = 'none'  # No features
        config['data']['sparsity'] = sparsity
        
        exp_name = f"lastfm_without_sparse{sparsity}"
        config['experiment_name'] = exp_name
        config['save_dir'] = f'results/lastfm_ablation/{exp_name}'
        
        filename = f"{exp_name}.yaml"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        generated_files.append(filepath)
        print(f"  ✓ {filename}")
    
    # 2. Models with features (add and concat) - for all sparsities and feature types
    models = [
        ('Model_userfeat_add', 'add'),
        ('Model_userfeat_concat', 'concat'),
        ('Model_userfeat_mtl', 'mtl')
    ]
    
    for model_name, model_short in models:
        for feat_type in feature_types:
            for sparsity in sparsities:
                config = create_base_lastfm_config()
                
                config['model']['name'] = model_name
                config['data']['feature_type'] = feat_type
                config['data']['sparsity'] = sparsity
                
                exp_name = f"lastfm_{model_short}_{feat_type}_sparse{sparsity}"
                config['experiment_name'] = exp_name
                config['save_dir'] = f'results/lastfm_ablation/{exp_name}'
                
                filename = f"{exp_name}.yaml"
                filepath = os.path.join(output_dir, filename)
                
                with open(filepath, 'w') as f:
                    yaml.dump(config, f, default_flow_style=False, sort_keys=False)
                
                generated_files.append(filepath)
                print(f"  ✓ {filename}")
    
    print(f"\n✓ Generated {len(generated_files)} Last.fm configs in {output_dir}")
    return generated_files



def print_lastfm_instructions():
    """Print instructions for running Last.fm experiments"""
    print("\n" + "="*70)
    print("INSTRUCTIONS TO RUN LAST.FM EXPERIMENTS")
    print("="*70)
    
    print("\n1. Run all Last.fm experiments:")
    print("   python run_lastfm_experiments.py config/experiments_lastfm/ablation/")
    
    print("\n2. This will run 28 experiments:")
    print("   - 4 sparse levels for baseline (Model_without)")
    print("   - 2 models × 3 feature types × 4 sparsities = 24 experiments")
    print("   = 28 experiments × 3 runs each = 84 training runs")
    
    print("\n3. Sparsities tested: 5th, 30th, 60th, 100th percentile")
    print("   Feature types: real, random, shuffled")
    
    print("\n4. Each experiment runs 3 times with different seeds")
    
    print("\n5. Analyze results:")
    print("   python analyze_lastfm_results.py")
    
    print("\n6. Results will be saved to:")
    print("   - Individual: results/lastfm_ablation/<exp_name>/")
    print("   - Summary: results/lastfm_ablation_summary.csv")
    
    print("\n" + "="*70)
    print("ESTIMATED TIME")
    print("="*70)
    print("If each experiment takes ~15 minutes:")
    print("Total time: 28 × 15 = 420 minutes (~7 hours)")
    print("="*70)


if __name__ == "__main__":
    print("="*70)
    print("GENERATING LAST.FM EXPERIMENT CONFIGS")
    print("="*70)
    
    # Generate configs
    generated_files = generate_lastfm_ablation_configs()
    
    # Print instructions
    print_lastfm_instructions()
    
    print("\n✓ Setup complete! Ready to run Last.fm experiments.")