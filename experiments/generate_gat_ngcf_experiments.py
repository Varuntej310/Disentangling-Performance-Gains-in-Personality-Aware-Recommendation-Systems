"""
Generate experiment configs for GAT and NGCF models

Tests these new architectures with personality integration:
- GAT: Graph Attention Networks (attention-based aggregation)
- NGCF: Neural Graph Collaborative Filtering (explicit high-order modeling)

Compares with GraphSAGE baseline
"""
import os
import yaml
from pathlib import Path


def create_base_config():
    """Base configuration template"""
    return {
        'paths': {
            'ratings': "/kaggle/input/personality-2018/personality-isf2018/ratings.csv",
            'personality': "/kaggle/input/personality-2018/personality-isf2018/personality-data.csv",
            'movies': "/kaggle/input/movielens-25m/ml-25m/movies.csv"
        },
        'data': {
            'name': 'movielens',
            'sparsity_percentile': 1.0,
            'min_interactions': 2,
            'personality_columns': ['openness', 'agreeableness', 'emotional_stability',
                                   'conscientiousness', 'extraversion']
        },
        'model': {
            'hidden_channels': 5,
            'num_neighbors': [20, 10],
            'personality_loss_weight': 0.0
        },
        'training': {
            'num_epochs': 50,
            'batch_size': 128,
            'lr': 0.001,
            'patience': 7,
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


def generate_architecture_comparison(output_dir="config/experiments/architecture_comparison"):
    """
    Compare GraphSAGE, GAT, and NGCF with different personality integration methods
    
    For each architecture test:
    - Linear integration
    - Concatenation integration
    - Multi-task learning
    - Architecture-specific variant
    """
    os.makedirs(output_dir, exist_ok=True)
    generated_files = []
    
    print("="*70)
    print("Generating Architecture Comparison Experiments")
    print("="*70)
    
    # Define architectures and their variants
    architectures = [
        # GraphSAGE (baseline)
        ('Model_linear', 'graphsage_linear', 0.0),
        ('Model_concat', 'graphsage_concat', 0.0),
        ('Model_mtl', 'graphsage_mtl', 0.5),
        
        # GAT
        ('GAT_linear', 'gat_linear', 0.0),
        ('GAT_concat', 'gat_concat', 0.0),
        ('GAT_mtl', 'gat_mtl', 0.5),
        ('GAT_attention_guided', 'gat_attention_guided', 0.0),
        
        # NGCF
        ('NGCF_linear', 'ngcf_linear', 0.0),
        ('NGCF_concat', 'ngcf_concat', 0.0),
        ('NGCF_mtl', 'ngcf_mtl', 0.5),
        ('NGCF_propagation_aware', 'ngcf_propagation_aware', 0.0),
    ]
    
    for model_name, exp_name, lambda_weight in architectures:
        config = create_base_config()
        
        config['model']['name'] = model_name
        config['model']['personality_loss_weight'] = lambda_weight
        
        # Use real personality
        config['data']['personality_type'] = 'real'
        config['data']['noise_distribution'] = None
        
        config['experiment_name'] = exp_name
        config['save_dir'] = f"results/architecture_comparison/{exp_name}"
        
        # Save
        filename = f"{exp_name}.yaml"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        generated_files.append(filepath)
        print(f"  ✓ {filename}")
    
    print(f"\n✓ Generated {len(generated_files)} configs in {output_dir}")
    return generated_files


def generate_gat_ablation(output_dir="config/experiments/gat_ablation"):
    """
    Ablation study for GAT models
    Test with: real, shuffled, noise, and no personality
    """
    os.makedirs(output_dir, exist_ok=True)
    generated_files = []
    
    print("\n" + "="*70)
    print("Generating GAT Ablation Experiments")
    print("="*70)
    
    gat_models = [
        ('GAT_linear', 0.0),
        ('GAT_concat', 0.0),
        ('GAT_mtl', 0.5),
    ]
    
    personality_types = [
        ('real', None, 'real'),
        ('shuffled', None, 'shuffled'),
        ('noise', 'uniform', 'noise'),
    ]
    
    for model_name, lambda_weight in gat_models:
        for pers_type, noise_dist, suffix in personality_types:
            config = create_base_config()
            
            config['model']['name'] = model_name
            config['model']['personality_loss_weight'] = lambda_weight
            
            config['data']['personality_type'] = pers_type
            config['data']['noise_distribution'] = noise_dist
            
            model_short = model_name.replace('GAT_', '').lower()
            exp_name = f"gat_{model_short}_{suffix}"
            
            config['experiment_name'] = exp_name
            config['save_dir'] = f"results/gat_ablation/{exp_name}"
            
            filename = f"{exp_name}.yaml"
            filepath = os.path.join(output_dir, filename)
            
            with open(filepath, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            
            generated_files.append(filepath)
            print(f"  ✓ {filename}")
    
    print(f"\n✓ Generated {len(generated_files)} configs in {output_dir}")
    return generated_files


def generate_ngcf_ablation(output_dir="config/experiments/ngcf_ablation"):
    """
    Ablation study for NGCF models
    Test with: real, shuffled, noise, and no personality
    """
    os.makedirs(output_dir, exist_ok=True)
    generated_files = []
    
    print("\n" + "="*70)
    print("Generating NGCF Ablation Experiments")
    print("="*70)
    
    ngcf_models = [
        ('NGCF_linear', 0.0),
        ('NGCF_concat', 0.0),
        ('NGCF_mtl', 0.5),
    ]
    
    personality_types = [
        ('real', None, 'real'),
        ('shuffled', None, 'shuffled'),
        ('noise', 'uniform', 'noise'),
    ]
    
    for model_name, lambda_weight in ngcf_models:
        for pers_type, noise_dist, suffix in personality_types:
            config = create_base_config()
            
            config['model']['name'] = model_name
            config['model']['personality_loss_weight'] = lambda_weight
            
            config['data']['personality_type'] = pers_type
            config['data']['noise_distribution'] = noise_dist
            
            model_short = model_name.replace('NGCF_', '').lower()
            exp_name = f"ngcf_{model_short}_{suffix}"
            
            config['experiment_name'] = exp_name
            config['save_dir'] = f"results/ngcf_ablation/{exp_name}"
            
            filename = f"{exp_name}.yaml"
            filepath = os.path.join(output_dir, filename)
            
            with open(filepath, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            
            generated_files.append(filepath)
            print(f"  ✓ {filename}")
    
    print(f"\n✓ Generated {len(generated_files)} configs in {output_dir}")
    return generated_files


def generate_cross_architecture_ablation(output_dir="config/experiments/cross_architecture"):
    """
    Compare all three architectures (GraphSAGE, GAT, NGCF) head-to-head
    across different sparsity levels
    """
    os.makedirs(output_dir, exist_ok=True)
    generated_files = []
    
    print("\n" + "="*70)
    print("Generating Cross-Architecture Comparison")
    print("="*70)
    
    models = [
        ('Model_linear', 'sage', 0.0),
        ('GAT_linear', 'gat', 0.0),
        ('NGCF_linear', 'ngcf', 0.0),
    ]
    
    sparsity_levels = [1, 30, 60, 100]
    
    for model_name, arch_name, lambda_weight in models:
        for sparsity in sparsity_levels:
            config = create_base_config()
            
            config['model']['name'] = model_name
            config['model']['personality_loss_weight'] = lambda_weight
            
            config['data']['sparsity_percentile'] = float(sparsity)
            config['data']['personality_type'] = 'real'
            config['data']['noise_distribution'] = None
            
            exp_name = f"{arch_name}_sp{sparsity}"
            config['experiment_name'] = exp_name
            config['save_dir'] = f"results/cross_architecture/{exp_name}"
            
            filename = f"{exp_name}.yaml"
            filepath = os.path.join(output_dir, filename)
            
            with open(filepath, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            
            generated_files.append(filepath)
            print(f"  ✓ {filename}")
    
    print(f"\n✓ Generated {len(generated_files)} configs in {output_dir}")
    return generated_files


def print_summary(all_generated):
    """Print summary"""
    print("\n" + "="*70)
    print("GAT & NGCF EXPERIMENT GENERATION COMPLETE")
    print("="*70)
    
    total = sum(len(files) for files in all_generated.values())
    
    print(f"\nTotal configs generated: {total}\n")
    
    for exp_type, files in all_generated.items():
        print(f"  {exp_type}: {len(files)} configs")
    
    print("\n" + "="*70)
    print("HOW TO RUN")
    print("="*70)
    
    print("\n1. Run specific experiment group:")
    for exp_type in all_generated.keys():
        dir_name = exp_type.replace(' ', '_').lower()
        print(f"   python run_batch_experiments.py config/experiments/{dir_name}/")
    
    print("\n2. Run ALL GAT/NGCF experiments:")
    print("   python run_batch_experiments.py config/experiments/architecture_comparison/")
    print("   python run_batch_experiments.py config/experiments/gat_ablation/")
    print("   python run_batch_experiments.py config/experiments/ngcf_ablation/")
    print("   python run_batch_experiments.py config/experiments/cross_architecture/")
    
    print("\n3. Test single experiment:")
    print("   python run_experiment.py config/experiments/architecture_comparison/gat_linear.yaml")
    
    print("\n" + "="*70)
    print("WHAT YOU'LL GET")
    print("="*70)
    
    print("\n• Architecture Comparison: Which architecture is best?")
    print("  - GraphSAGE vs GAT vs NGCF")
    print("  - Different personality integration methods")
    
    print("\n• GAT Ablation: Does attention help with personality?")
    print("  - Real vs shuffled vs noise personality")
    print("  - Attention mechanism effectiveness")
    
    print("\n• NGCF Ablation: Do high-order paths matter?")
    print("  - Real vs shuffled vs noise personality")
    print("  - Collaborative filtering with personality")
    
    print("\n• Cross-Architecture: How do they scale?")
    print("  - Performance at different graph densities")
    print("  - 1%, 30%, 60%, 100% sparsity")
    
    print("\n" + "="*70)


def main():
    """Main generation function"""
    print("\n" + "="*70)
    print("GAT & NGCF EXPERIMENT CONFIG GENERATOR")
    print("="*70)
    print("\nThis will generate experiments for:")
    print("  1. Architecture comparison (11 experiments)")
    print("  2. GAT ablation (9 experiments)")
    print("  3. NGCF ablation (9 experiments)")
    print("  4. Cross-architecture (12 experiments)")
    print("\nTotal: 41 experiments")
    print("")
    
    all_generated = {}
    
    all_generated['Architecture Comparison'] = generate_architecture_comparison()
    all_generated['GAT Ablation'] = generate_gat_ablation()
    all_generated['NGCF Ablation'] = generate_ngcf_ablation()
    all_generated['Cross Architecture'] = generate_cross_architecture_ablation()
    
    print_summary(all_generated)
    
    print("\n✓ Setup complete! Ready to test new architectures.")


if __name__ == "__main__":
    main()