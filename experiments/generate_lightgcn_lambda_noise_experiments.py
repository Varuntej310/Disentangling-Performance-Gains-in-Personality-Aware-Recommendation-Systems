"""
Generate comprehensive experiment configs:
1. LightGCN across different sparsity levels
2. Lambda sweep for MTL model
3. Noise distribution sweep
"""
import os
import yaml
from pathlib import Path


def create_base_config():
    """Base configuration template"""
    return {
        'paths': {
            'ratings': "/home/fac/ashok.sairam/btp-varun/btp-1/ratings.csv",
            'personality': "/home/fac/ashok.sairam/btp-varun/btp-1/personality-data.csv",
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
            'personality_loss_weight': 0.0
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


def generate_lightgcn_sparsity_experiments(output_dir="config/experiments/lightgcn_sparsity"):
    """
    Generate LightGCN experiments across different sparsity levels
    
    Sparsity levels: 5%, 30%, 60%, 100%
    """
    os.makedirs(output_dir, exist_ok=True)

    sparsity_levels = [5, 30, 60, 100]
    generated_files = []
    
    print("="*70)
    print("Generating LightGCN Sparsity Experiments")
    print("="*70)
    
    for sparsity in sparsity_levels:
        config = create_base_config()
        
        # LightGCN specific settings
        config['model'] = {
            'name': 'LightGCN',
            'embedding_dim': 5,
            'num_layers': 2
        }
        
        # Set sparsity
        config['data']['sparsity_percentile'] = float(sparsity)
        config['data']['personality_type'] = 'real'  # LightGCN doesn't use personality
        config['data']['noise_distribution'] = None
        
        # LightGCN uses higher learning rate
        config['training']['lr'] = 0.01
        
        # Experiment naming
        exp_name = f"lightgcn_sp{sparsity}"
        config['experiment_name'] = exp_name
        config['save_dir'] = f"results/lightgcn_sparsity/{exp_name}"
        
        # Save config
        filename = f"{exp_name}.yaml"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        generated_files.append(filepath)
        print(f"  ✓ {filename}")
    
    print(f"\n✓ Generated {len(generated_files)} LightGCN configs in {output_dir}")
    return generated_files


def generate_lambda_sweep_experiments(output_dir="config/experiments/lambda_sweep"):
    """
    Generate lambda sweep experiments for MTL model
    
    Lambda values: 0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 10
    Sparsity: 5% (default)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    lambda_values = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 10]
    generated_files = []
    
    print("\n" + "="*70)
    print("Generating Lambda Sweep Experiments (MTL)")
    print("="*70)
    
    for lam in lambda_values:
        config = create_base_config()
        
        # MTL model settings
        config['model']['name'] = 'Model_mtl'
        config['model']['personality_loss_weight'] = lam
        
        # Use real personality at 1% sparsity
        config['data']['sparsity_percentile'] = 5.0
        config['data']['personality_type'] = 'real'
        config['data']['noise_distribution'] = None
        
        # Experiment naming
        exp_name = f"mtl_lambda_{lam}"
        config['experiment_name'] = exp_name
        config['save_dir'] = f"results/lambda_sweep/{exp_name}"
        
        # Save config
        filename = f"lambda_{lam}.yaml"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        generated_files.append(filepath)
        print(f"  ✓ {filename}")
    
    print(f"\n✓ Generated {len(generated_files)} lambda sweep configs in {output_dir}")
    return generated_files


def generate_noise_distribution_experiments(output_dir="config/experiments/noise_distributions"):
    """
    Generate noise distribution experiments
    
    Distributions: uniform, normal, laplace, bernoulli, exponential
    Models: Linear
    Sparsity: 5% (default)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    distributions = ['uniform', 'normal', 'laplace', 'bernoulli', 'exponential']
    models = ['Model_linear']
    generated_files = []
    
    print("\n" + "="*70)
    print("Generating Noise Distribution Experiments")
    print("="*70)
    
    for model in models:
        for dist in distributions:
            config = create_base_config()
            
            # Model settings
            config['model']['name'] = model
            if model == 'Model_mtl':
                config['model']['personality_loss_weight'] = 0.5
            
            # Noise distribution settings
            config['data']['sparsity_percentile'] = 5.0
            config['data']['personality_type'] = 'noise'
            config['data']['noise_distribution'] = dist
            
            # Experiment naming
            model_short = model.replace('Model_', '').lower()
            exp_name = f"{model_short}_noise_{dist}"
            config['experiment_name'] = exp_name
            config['save_dir'] = f"results/noise_distributions/{exp_name}"
            
            # Save config
            filename = f"{exp_name}.yaml"
            filepath = os.path.join(output_dir, filename)
            
            with open(filepath, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            
            generated_files.append(filepath)
            print(f"  ✓ {filename}")
    
    print(f"\n✓ Generated {len(generated_files)} noise distribution configs in {output_dir}")
    return generated_files



def print_summary(all_generated):
    """Print summary of all generated configs"""
    print("\n" + "="*70)
    print("GENERATION COMPLETE - SUMMARY")
    print("="*70)
    
    total = sum(len(files) for files in all_generated.values())
    
    print(f"\nTotal configs generated: {total}\n")
    
    for exp_type, files in all_generated.items():
        print(f"  {exp_type}: {len(files)} configs")
    
    print("\n" + "="*70)
    print("HOW TO RUN")
    print("="*70)
    
    print("\n1. Run specific experiment type:")
    for exp_type in all_generated.keys():
        dir_name = exp_type.replace(' ', '_').lower()
        print(f"   python run_batch_experiments.py config/experiments/{dir_name}/")
    
    print("\n2. Run ALL experiments:")
    print("   python run_all_comprehensive_experiments.py")
    
    print("\n3. Run with parallel GPUs (faster):")
    print("   # See run_parallel_experiments.sh")
    
    print("\n" + "="*70)
    print("TIME ESTIMATES")
    print("="*70)
    
    estimates = {
        'LightGCN Sparsity': (5, 10),
        'Lambda Sweep': (7, 10),
        'Noise Distributions': (15, 10),
        'Model Comparison': (4, 10),
        'Sparsity Sweep All': (20, 10),
    }
    
    total_time = 0
    for exp_type, (count, mins) in estimates.items():
        time = count * mins
        total_time += time
        print(f"  {exp_type}: {count} exp × {mins} min = {time} min (~{time/60:.1f}h)")
    
    print(f"\n  TOTAL: ~{total_time} min (~{total_time/60:.1f}h)")
    print(f"  With 3 GPUs: ~{total_time/3/60:.1f}h")
    
    print("\n" + "="*70)


def generate_run_all_script():
    """Generate a script to run all experiments"""
    script_content = '''#!/usr/bin/env python
"""
Run all comprehensive experiments sequentially
"""
import sys
from run_batch_experiments import run_batch_experiments

experiments = [
    ('config/experiments/lightgcn_sparsity', 'LightGCN Sparsity'),
    ('config/experiments/lambda_sweep', 'Lambda Sweep'),
    ('config/experiments/noise_distributions', 'Noise Distributions'),
    ('config/experiments/model_comparison', 'Model Comparison'),
    ('config/experiments/sparsity_sweep', 'Sparsity Sweep All Models'),
]

print("="*70)
print("RUNNING ALL COMPREHENSIVE EXPERIMENTS")
print("="*70)
print(f"Total experiment groups: {len(experiments)}\\n")

for i, (config_dir, name) in enumerate(experiments, 1):
    print(f"\\n{'#'*70}")
    print(f"# GROUP {i}/{len(experiments)}: {name}")
    print(f"{'#'*70}\\n")
    
    try:
        run_batch_experiments(config_dir, "*.yaml")
    except Exception as e:
        print(f"\\n❌ Error in {name}: {e}")
        continue

print("\\n" + "="*70)
print("ALL EXPERIMENTS COMPLETE!")
print("="*70)
'''
    
    with open('run_all_comprehensive_experiments.py', 'w') as f:
        f.write(script_content)
    
    os.chmod('run_all_comprehensive_experiments.py', 0o755)
    print("\n✓ Created: run_all_comprehensive_experiments.py")


def generate_parallel_script():
    """Generate bash script for parallel execution"""
    script_content = '''#!/bin/bash
# Run experiments in parallel across multiple GPUs

echo "=================================================="
echo "PARALLEL EXPERIMENT EXECUTION"
echo "=================================================="
echo ""
echo "This will use 3 GPUs to run experiments in parallel"
echo "Make sure you have 3 GPUs available!"
echo ""

# GPU 0: LightGCN and Lambda Sweep
CUDA_VISIBLE_DEVICES=0 python run_batch_experiments.py config/experiments/lightgcn_sparsity/ > logs/gpu0_lightgcn.log 2>&1 &
PID1=$!

# GPU 1: Noise distributions
CUDA_VISIBLE_DEVICES=1 python run_batch_experiments.py config/experiments/noise_distributions/ > logs/gpu1_noise.log 2>&1 &
PID2=$!

# GPU 2: Sparsity sweep
CUDA_VISIBLE_DEVICES=2 python run_batch_experiments.py config/experiments/sparsity_sweep/ > logs/gpu2_sparsity.log 2>&1 &
PID3=$!

echo "Started 3 parallel processes:"
echo "  GPU 0 (PID $PID1): LightGCN Sparsity"
echo "  GPU 1 (PID $PID2): Noise Distributions"
echo "  GPU 2 (PID $PID3): Sparsity Sweep"
echo ""
echo "Logs:"
echo "  tail -f logs/gpu0_lightgcn.log"
echo "  tail -f logs/gpu1_noise.log"
echo "  tail -f logs/gpu2_sparsity.log"
echo ""

# Wait for all to complete
wait $PID1
wait $PID2
wait $PID3

echo ""
echo "=================================================="
echo "ALL PARALLEL EXPERIMENTS COMPLETE!"
echo "=================================================="

# Run lambda sweep and model comparison on GPU 0 (quick ones)
echo ""
echo "Running remaining experiments..."
CUDA_VISIBLE_DEVICES=0 python run_batch_experiments.py config/experiments/lambda_sweep/
CUDA_VISIBLE_DEVICES=0 python run_batch_experiments.py config/experiments/model_comparison/

echo ""
echo "✓ EVERYTHING COMPLETE!"
'''
    
    os.makedirs('logs', exist_ok=True)
    
    with open('run_parallel_experiments.sh', 'w') as f:
        f.write(script_content)
    
    os.chmod('run_parallel_experiments.sh', 0o755)
    print("✓ Created: run_parallel_experiments.sh")


def main():
    """Main generation function"""
    print("\n" + "="*70)
    print("COMPREHENSIVE EXPERIMENT CONFIG GENERATOR")
    print("="*70)
    print("\nThis will generate configs for:")
    print("  1. LightGCN across different sparsity levels")
    print("  2. Lambda sweep for MTL model")
    print("  3. Noise distribution experiments")
    print("  4. Model comparison")
    print("  5. Sparsity sweep for all models")
    print("")
    
    all_generated = {}
    
    # Generate each experiment type
    all_generated['LightGCN Sparsity'] = generate_lightgcn_sparsity_experiments()
    all_generated['Lambda Sweep'] = generate_lambda_sweep_experiments()
    all_generated['Noise Distributions'] = generate_noise_distribution_experiments()
    # Generate helper scripts
    generate_run_all_script()
    generate_parallel_script()
    
    # Print summary
    print_summary(all_generated)
    
    print("\n✓ Setup complete! Ready to run experiments.")


if __name__ == "__main__":
    main()