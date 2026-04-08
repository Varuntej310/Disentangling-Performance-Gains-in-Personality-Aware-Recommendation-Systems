#!/usr/bin/env python
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
print(f"Total experiment groups: {len(experiments)}\n")

for i, (config_dir, name) in enumerate(experiments, 1):
    print(f"\n{'#'*70}")
    print(f"# GROUP {i}/{len(experiments)}: {name}")
    print(f"{'#'*70}\n")
    
    try:
        run_batch_experiments(config_dir, "*.yaml")
    except Exception as e:
        print(f"\n❌ Error in {name}: {e}")
        continue

print("\n" + "="*70)
print("ALL EXPERIMENTS COMPLETE!")
print("="*70)
