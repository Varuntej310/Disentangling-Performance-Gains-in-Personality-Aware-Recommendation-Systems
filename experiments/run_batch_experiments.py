"""
Batch experiment runner for multiple YAML configs
"""
import os
import sys
import glob
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from experiments.run_experiment import run_single_experiment


def run_batch_experiments(config_dir: str, pattern: str = "*.yaml"):
    """
    Run multiple experiments from a directory of YAML configs
    
    Args:
        config_dir: Directory containing YAML config files
        pattern: Glob pattern for config files (default: *.yaml)
    """
    # Find all config files
    config_pattern = os.path.join(config_dir, pattern)
    config_files = sorted(glob.glob(config_pattern))
    
    if not config_files:
        print(f"No config files found matching: {config_pattern}")
        sys.exit(1)
    
    print("="*70)
    print(f"BATCH EXPERIMENT RUNNER")
    print("="*70)
    print(f"Found {len(config_files)} config files:")
    for i, cfg in enumerate(config_files, 1):
        print(f"  {i}. {os.path.basename(cfg)}")
    print("="*70)
    
    # Run all experiments
    all_results = []
    
    for i, config_path in enumerate(config_files, 1):
        print(f"\n{'#'*70}")
        print(f"# EXPERIMENT {i}/{len(config_files)}: {os.path.basename(config_path)}")
        print(f"{'#'*70}\n")
        
        try:
            results = run_single_experiment(config_path)
            all_results.append(results)
        except Exception as e:
            print(f"\nERROR in experiment {i}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    # Create summary
    print("\n" + "="*70)
    print("BATCH SUMMARY")
    print("="*70)
    
    summary_data = []
    for res in all_results:
        row = {
            'experiment': res['experiment_name'],
            'model': res['model'],
            'num_runs': res['num_runs'],
        }
        row.update(res['avg_metrics'])
        row.update(res['std_metrics'])
        summary_data.append(row)
    
    summary_df = pd.DataFrame(summary_data)
    
    # Print summary table
    print("\nResults Summary:")
    print(summary_df[['experiment', 'model', 'HR@10', 'NDCG@10']].to_string(index=False))
    
    # Save summary
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = f"results/batch_summary_{timestamp}.csv"
    os.makedirs("results", exist_ok=True)
    summary_df.to_csv(summary_path, index=False)
    
    print(f"\nSummary saved to: {summary_path}")
    print("="*70)
    
    return summary_df


def generate_lambda_sweep_configs(
    base_config_path: str,
    lambdas: list,
    output_dir: str = "config/experiments/lambda_sweep"
):
    """
    Generate multiple config files for lambda sweep
    
    Args:
        base_config_path: Path to base YAML config
        lambdas: List of lambda values to sweep
        output_dir: Directory to save generated configs
    """
    from config.experiment_config import ExperimentConfig
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load base config
    base_config = ExperimentConfig.from_yaml(base_config_path)
    
    generated_files = []
    
    for lam in lambdas:
        # Modify config
        config = ExperimentConfig.from_dict(base_config.to_dict())
        config.experiment_name = f"lambda_{lam}"
        config.model.personality_loss_weight = lam
        config.save_dir = f"results/lambda_sweep/lambda_{lam}"
        
        # Save
        output_path = os.path.join(output_dir, f"lambda_{lam}.yaml")
        config.to_yaml(output_path)
        generated_files.append(output_path)
    
    print(f"Generated {len(generated_files)} config files in {output_dir}")
    return generated_files


def generate_noise_sweep_configs(
    base_config_path: str,
    distributions: list,
    output_dir: str = "config/experiments/noise_sweep"
):
    """
    Generate multiple config files for noise distribution sweep
    
    Args:
        base_config_path: Path to base YAML config
        distributions: List of noise distributions
        output_dir: Directory to save generated configs
    """
    from config.experiment_config import ExperimentConfig
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load base config
    base_config = ExperimentConfig.from_yaml(base_config_path)
    
    generated_files = []
    
    for dist in distributions:
        # Modify config
        config = ExperimentConfig.from_dict(base_config.to_dict())
        config.experiment_name = f"noise_{dist}"
        config.data.personality_type = "noise"
        config.data.noise_distribution = dist
        config.save_dir = f"results/noise_sweep/{dist}"
        
        # Save
        output_path = os.path.join(output_dir, f"noise_{dist}.yaml")
        config.to_yaml(output_path)
        generated_files.append(output_path)
    
    print(f"Generated {len(generated_files)} config files in {output_dir}")
    return generated_files


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python run_batch_experiments.py <config_dir> [pattern]")
        print("  python run_batch_experiments.py generate_lambda <base_config>")
        print("  python run_batch_experiments.py generate_noise <base_config>")
        print("\nExamples:")
        print("  python run_batch_experiments.py config/experiments/")
        print("  python run_batch_experiments.py config/experiments/ 'mtl_*.yaml'")
        print("  python run_batch_experiments.py generate_lambda config/experiments/baseline.yaml")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "generate_lambda":
        base_config = sys.argv[2] if len(sys.argv) > 2 else "config/experiments/baseline.yaml"
        lambdas = [0.0, 0.25, 0.5, 0.75, 1.0]
        generate_lambda_sweep_configs(base_config, lambdas)
    
    elif command == "generate_noise":
        base_config = sys.argv[2] if len(sys.argv) > 2 else "config/experiments/baseline.yaml"
        distributions = ["uniform", "normal", "laplace", "bernoulli", "exponential"]
        generate_noise_sweep_configs(base_config, distributions)
    
    else:
        # Run batch experiments
        config_dir = command
        pattern = sys.argv[2] if len(sys.argv) > 2 else "*.yaml"
        run_batch_experiments(config_dir, pattern)