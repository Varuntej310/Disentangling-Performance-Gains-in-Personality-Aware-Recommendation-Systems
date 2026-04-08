"""
Format ablation results into LaTeX table format
"""
import pandas as pd
import numpy as np
import glob
from pathlib import Path


def load_latest_summary():
    """Load the most recent ablation summary CSV"""
    summaries = glob.glob("results/ablation_table_summary_*.csv")
    
    if not summaries:
        print("❌ No summary CSV found. Run experiments first.")
        return None
    
    latest = max(summaries, key=lambda x: Path(x).stat().st_mtime)
    print(f"Loading: {latest}")
    
    return pd.read_csv(latest)


def format_value(val):
    """Format a single value for LaTeX table"""
    if pd.isna(val) or val == 0:
        return "0.XXX"
    return f"{val:.4f}"


def generate_latex_table(df, output_file="results/ablation_table.tex"):
    """
    Generate complete LaTeX table matching the paper format
    """
    
    # Table structure
    models = [
        ('linear', 'real', 'Linear + org.\\ personality'),
        ('linear', 'shuffled', 'Linear + shuffled personality'),
        ('linear', 'noise', 'Linear + uniform noise'),
        ('concat', 'real', 'Concatenation + org.\\ personality'),
        ('concat', 'shuffled', 'Concatenation + shuffled personality'),
        ('concat', 'noise', 'Concatenation + uniform noise'),
        ('mtl', 'real', 'MTL + org.\\ personality'),
        ('mtl', 'shuffled', 'MTL + shuffled personality'),
        ('mtl', 'noise', 'MTL + uniform noise'),
    ]
    
    # Start building LaTeX
    latex_lines = []
    
    # Table header
    latex_lines.extend([
        "\\begin{table*}[htbp]",
        "\\centering",
        "\\small",
        "\\renewcommand{\\arraystretch}{1.05}",
        "\\caption{Ablation results for different graph densities. Values: HR@5, NDCG@5, HR@10, NDCG@10.}",
        "\\label{tab:ablation_graph_density}",
        "\\begin{tabular}{@{} l l",
        "  S[table-format=1.4] S[table-format=1.4] S[table-format=1.4] S[table-format=1.4]",
        "  S[table-format=1.4] S[table-format=1.4] S[table-format=1.4] S[table-format=1.4] @{}}",
        "\\toprule",
        "\\multirow{2}{*}{Model} & \\multirow{2}{*}{Metric}",
        " & \\multicolumn{4}{c}{\\textbf{With Personality}} ",
        " & \\multicolumn{4}{c}{\\textbf{Zeroed Personality}} \\\\",
        "\\cmidrule(lr){3-6} \\cmidrule(lr){7-10}",
        " & & {\\footnotesize 5\\%} & {\\footnotesize 30\\%} & {\\footnotesize 60\\%} & {\\footnotesize 100\\%}",
        " & {\\footnotesize 5\\%} & {\\footnotesize 30\\%} & {\\footnotesize 60\\%} & {\\footnotesize 100\\%} \\\\",
        "\\midrule",
    ])
    
    # Data rows
    for i, (model_key, pers_key, model_label) in enumerate(models):
        # Get subset for this model+personality combo
        subset = df[
            (df['model'].str.contains(model_key, case=False)) & 
            (df['personality'] == pers_key)
        ]
        
        if len(subset) == 0:
            continue
        
        # Add spacing between model groups
        if i > 0 and i % 3 == 0:
            latex_lines.append("\\midrule")
        elif i % 3 == 1:
            latex_lines.append("\\addlinespace")
        
        # Four metrics per model
        for j, metric in enumerate(['HR@5', 'NDCG@5', 'HR@10', 'NDCG@10']):
            if j == 0:
                # First row includes model label
                row_start = f"\\multirow{{4}}{{*}}{{{model_label}}}"
            else:
                row_start = ""
            
            # Metric name
            parts = [row_start, f" & {metric:7s}"]
            
            # Values with personality (1%, 30%, 60%, 100%)
            for sp in [5, 30, 60, 100]:
                row = subset[subset['sparsity'] == sp]
                if len(row) > 0 and metric in row.columns:
                    val = row[metric].values[0]
                    parts.append(f" & {format_value(val)}")
                else:
                    parts.append(" & 0.XXX")
            
            # Values with zeroed personality
            for sp in [5, 30, 60, 100]:
                row = subset[subset['sparsity'] == sp]
                zeroed_col = f"{metric}_zeroed"
                if len(row) > 0 and zeroed_col in row.columns:
                    val = row[zeroed_col].values[0]
                    parts.append(f" & {format_value(val)}")
                else:
                    parts.append(" & 0.XXX")
            
            parts.append(" \\\\")
            latex_lines.append("".join(parts))
    
    # Table footer
    latex_lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table*}",
    ])
    
    # Write to file
    with open(output_file, 'w') as f:
        f.write("\n".join(latex_lines))
    
    print(f"✓ LaTeX table saved to: {output_file}")
    
    # Also print to console
    print("\n" + "="*100)
    print("LATEX TABLE")
    print("="*100)
    for line in latex_lines:
        print(line)
    print("="*100)


def generate_simple_table(df, output_file="results/ablation_table_simple.txt"):
    """
    Generate a simple text table for quick viewing
    """
    
    models = [
        ('linear', 'real', 'Linear + org.'),
        ('linear', 'shuffled', 'Linear + shuf.'),
        ('linear', 'noise', 'Linear + noise'),
        ('concat', 'real', 'Concat + org.'),
        ('concat', 'shuffled', 'Concat + shuf.'),
        ('concat', 'noise', 'Concat + noise'),
        ('mtl', 'real', 'MTL + org.'),
        ('mtl', 'shuffled', 'MTL + shuf.'),
        ('mtl', 'noise', 'MTL + noise'),
    ]
    
    lines = []
    lines.append("="*140)
    lines.append("ABLATION TABLE RESULTS")
    lines.append("="*140)
    lines.append("")
    lines.append(f"{'Model':<25} {'Metric':<10} {'--- With Personality ---':>50} {'--- Zeroed Personality ---':>50}")
    lines.append(f"{'':25} {'':10} {'5%':>10} {'30%':>10} {'60%':>10} {'100%':>10} {'5%':>10} {'30%':>10} {'60%':>10} {'100%':>10}")
    lines.append("-"*140)
    
    for model_key, pers_key, model_label in models:
        subset = df[
            (df['model'].str.contains(model_key, case=False)) & 
            (df['personality'] == pers_key)
        ]
        
        if len(subset) == 0:
            continue
        
        for metric in ['HR@5', 'NDCG@5', 'HR@10', 'NDCG@10']:
            parts = [f"{model_label:<25}", f"{metric:<10}"]
            
            # With personality
            for sp in [5, 30, 60, 100]:
                row = subset[subset['sparsity'] == sp]
                if len(row) > 0 and metric in row.columns:
                    val = row[metric].values[0]
                    if not pd.isna(val):
                        parts.append(f"{val:>10.4f}")
                    else:
                        parts.append(f"{'---':>10}")
                else:
                    parts.append(f"{'---':>10}")
            
            # Zeroed personality
            for sp in [5, 30, 60, 100]:
                row = subset[subset['sparsity'] == sp]
                zeroed_col = f"{metric}_zeroed"
                if len(row) > 0 and zeroed_col in row.columns:
                    val = row[zeroed_col].values[0]
                    if not pd.isna(val):
                        parts.append(f"{val:>10.4f}")
                    else:
                        parts.append(f"{'---':>10}")
                else:
                    parts.append(f"{'---':>10}")
            
            lines.append("".join(parts))
        
        lines.append("")
    
    lines.append("="*140)
    
    # Write to file
    with open(output_file, 'w') as f:
        f.write("\n".join(lines))
    
    print(f"✓ Simple table saved to: {output_file}")
    
    # Print to console
    for line in lines:
        print(line)


def main():
    """Main formatting function"""
    print("="*70)
    print("FORMATTING ABLATION RESULTS")
    print("="*70)
    
    # Load data
    df = load_latest_summary()
    
    if df is None:
        return
    
    print(f"\nLoaded {len(df)} experiment results")
    print(f"Models: {df['model'].unique()}")
    print(f"Personality types: {df['personality'].unique()}")
    print(f"Sparsity levels: {sorted(df['sparsity'].unique())}")
    
    # Generate LaTeX table
    print("\n" + "-"*70)
    generate_latex_table(df)
    
    # Generate simple text table
    print("\n" + "-"*70)
    generate_simple_table(df)
    
    print("\n" + "="*70)
    print("✓ FORMATTING COMPLETE")
    print("="*70)
    print("\nOutput files:")
    print("  - results/ablation_table.tex (LaTeX)")
    print("  - results/ablation_table_simple.txt (Text)")
    print("  - results/ablation_table_summary_*.csv (Raw data)")


if __name__ == "__main__":
    main()