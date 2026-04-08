"""
Analyze Last.fm experiment results
"""
import json
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sns.set_style("whitegrid")


def load_lastfm_results():
    """Load all Last.fm experiment results"""
    # Find latest summary CSV
    summaries = glob.glob("results/lastfm_ablation_summary_*.csv")
    
    if summaries:
        latest = max(summaries, key=lambda x: Path(x).stat().st_mtime)
        print(f"Loading: {latest}")
        return pd.read_csv(latest)
    
    # Otherwise, load from individual results
    result_files = glob.glob("results/lastfm_ablation/**/results.json", recursive=True)
    
    if not result_files:
        print("❌ No results found. Run experiments first.")
        return None
    
    all_results = []
    for result_file in result_files:
        with open(result_file) as f:
            data = json.load(f)
        
        row = {
            'experiment': data['experiment_name'],
            'model': data['model'],
            'feature_type': data['feature_type'],
            'num_runs': data['num_runs'],
        }
        row.update(data['avg_metrics'])
        row.update(data['std_metrics'])
        all_results.append(row)
    
    return pd.DataFrame(all_results)


def analyze_feature_ablation(df, output_dir="results/lastfm_analysis"):
    """Analyze feature ablation (real vs random)"""
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Filter models with features
    models_with_feat = df[df['model'] != 'Model_without'].copy()
    
    if len(models_with_feat) == 0:
        print("No feature ablation results found")
        return
    
    # Pivot for comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for idx, metric in enumerate(['HR@10', 'NDCG@10']):
        pivot = models_with_feat.pivot_table(
            values=metric,
            index='model',
            columns='feature_type',
            aggfunc='mean'
        )
        
        pivot.plot(kind='bar', ax=axes[idx], alpha=0.8)
        axes[idx].set_title(f'Feature Ablation: {metric}')
        axes[idx].set_xlabel('Model')
        axes[idx].set_ylabel(metric)
        axes[idx].legend(title='Feature Type')
        axes[idx].grid(True, alpha=0.3, axis='y')
        
        # Rotate x labels
        axes[idx].set_xticklabels(axes[idx].get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/feature_ablation.png", dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/feature_ablation.png")
    plt.close()
    
    # Print table
    print("\n" + "="*70)
    print("FEATURE ABLATION ANALYSIS")
    print("="*70)
    
    for model in models_with_feat['model'].unique():
        subset = models_with_feat[models_with_feat['model'] == model]
        print(f"\n{model}:")
        
        for metric in ['HR@10', 'NDCG@10']:
            real_val = subset[subset['feature_type'] == 'real'][metric].values
            rand_val = subset[subset['feature_type'] == 'random'][metric].values
            
            if len(real_val) > 0 and len(rand_val) > 0:
                delta = real_val[0] - rand_val[0]
                print(f"  {metric}: real={real_val[0]:.4f}, random={rand_val[0]:.4f}, "
                      f"delta={delta:+.4f}")


def compare_all_models(df, output_dir="results/lastfm_analysis"):
    """Compare all models including baseline"""
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Get best configuration for each model
    best_configs = []
    
    for model in df['model'].unique():
        subset = df[df['model'] == model]
        # For models with features, pick the best feature type
        best_idx = subset['NDCG@10'].idxmax()
        best_configs.append(subset.loc[best_idx])
    
    comparison = pd.DataFrame(best_configs)
    
    # Bar plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    x = np.arange(len(comparison))
    width = 0.35
    
    # HR and NDCG
    axes[0].bar(x - width/2, comparison['HR@5'], width, label='HR@5', alpha=0.8)
    axes[0].bar(x + width/2, comparison['HR@10'], width, label='HR@10', alpha=0.8)
    axes[0].set_xlabel('Model')
    axes[0].set_ylabel('Hit Rate')
    axes[0].set_title('Model Comparison: Hit Rate')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(comparison['model'].str.replace('Model_', ''), 
                            rotation=45, ha='right')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')
    
    axes[1].bar(x - width/2, comparison['NDCG@5'], width, label='NDCG@5', alpha=0.8)
    axes[1].bar(x + width/2, comparison['NDCG@10'], width, label='NDCG@10', alpha=0.8)
    axes[1].set_xlabel('Model')
    axes[1].set_ylabel('NDCG')
    axes[1].set_title('Model Comparison: NDCG')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(comparison['model'].str.replace('Model_', ''), 
                            rotation=45, ha='right')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/model_comparison.png", dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/model_comparison.png")
    plt.close()
    
    # Print table
    print("\n" + "="*70)
    print("MODEL COMPARISON")
    print("="*70)
    print(comparison[['model', 'feature_type', 'HR@10', 'NDCG@10']].to_string(index=False))


def generate_summary_report(df, output_file="results/lastfm_analysis/summary_report.txt"):
    """Generate text summary report"""
    
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        f.write("="*70 + "\n")
        f.write("LAST.FM EXPERIMENTS SUMMARY REPORT\n")
        f.write("="*70 + "\n\n")
        
        # Overall statistics
        f.write(f"Total experiments: {len(df)}\n")
        f.write(f"Models tested: {', '.join(df['model'].unique())}\n\n")
        
        # Best performer
        f.write("="*70 + "\n")
        f.write("BEST PERFORMER\n")
        f.write("="*70 + "\n\n")
        
        best_idx = df['NDCG@10'].idxmax()
        best = df.loc[best_idx]
        f.write(f"Model: {best['model']}\n")
        f.write(f"Feature Type: {best['feature_type']}\n")
        f.write(f"HR@10: {best['HR@10']:.4f}\n")
        f.write(f"NDCG@10: {best['NDCG@10']:.4f}\n\n")
        
        # Feature ablation
        f.write("="*70 + "\n")
        f.write("FEATURE ABLATION (Real vs Random)\n")
        f.write("="*70 + "\n\n")
        
        for model in df[df['model'] != 'Model_without']['model'].unique():
            subset = df[df['model'] == model]
            real = subset[subset['feature_type'] == 'real']
            rand = subset[subset['feature_type'] == 'random']
            
            if len(real) > 0 and len(rand) > 0:
                f.write(f"{model}:\n")
                delta = real['NDCG@10'].values[0] - rand['NDCG@10'].values[0]
                f.write(f"  NDCG@10: real={real['NDCG@10'].values[0]:.4f}, "
                       f"random={rand['NDCG@10'].values[0]:.4f}, "
                       f"delta={delta:+.4f}\n\n")
    
    print(f"✓ Saved: {output_file}")


def main():
    """Main analysis function"""
    print("="*70)
    print("LAST.FM RESULTS ANALYSIS")
    print("="*70)
    
    # Load results
    print("\n[1/4] Loading results...")
    df = load_lastfm_results()
    
    if df is None or len(df) == 0:
        print("❌ No results found. Run experiments first.")
        return
    
    print(f"Loaded {len(df)} experiments")
    
    # Analyze feature ablation
    print("\n[2/4] Analyzing feature ablation...")
    analyze_feature_ablation(df)
    
    # Compare models
    print("\n[3/4] Comparing models...")
    compare_all_models(df)
    
    # Generate summary
    print("\n[4/4] Generating summary report...")
    generate_summary_report(df)
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)
    print("\nOutput files:")
    print("  - results/lastfm_analysis/feature_ablation.png")
    print("  - results/lastfm_analysis/model_comparison.png")
    print("  - results/lastfm_analysis/summary_report.txt")


if __name__ == "__main__":
    main()