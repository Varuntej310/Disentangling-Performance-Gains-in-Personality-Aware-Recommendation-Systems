"""
Run all comprehensive experiments sequentially
"""
import sys
import os
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from run_batch_experiments import run_batch_experiments


def main():
    experiments = [
        ('config/experiments/lightgcn_sparsity', 'LightGCN Sparsity Sweep', 4),
        ('config/experiments/lambda_sweep', 'Lambda Sweep (MTL)', 7),
        ('config/experiments/noise_distributions', 'Noise Distribution Tests', 5),
    ]
    
    total_experiments = sum(count for _, _, count in experiments)
    
    print("="*80)
    print("COMPREHENSIVE EXPERIMENTS - BATCH RUN")
    print("="*80)
    print(f"\nTotal experiment groups: {len(experiments)}")
    print(f"Total experiments: {total_experiments}")
    print(f"Estimated time: ~{total_experiments * 10} minutes (~{total_experiments * 10 / 60:.1f} hours)")
    print("\nExperiment groups:")
    for i, (_, name, count) in enumerate(experiments, 1):
        print(f"  {i}. {name}: {count} experiments")
    
    print("\n" + "="*80)
    
    response = input("\nProceed with all experiments? [y/N]: ")
    if response.lower() not in ['y', 'yes']:
        print("Aborted by user.")
        return
    
    print("\nStarting experiments...\n")
    
    start_time = datetime.now()
    results_summary = []
    failed_groups = []
    
    for i, (config_dir, name, expected_count) in enumerate(experiments, 1):
        print("\n" + "#"*80)
        print(f"# GROUP {i}/{len(experiments)}: {name}")
        print(f"# Expected: {expected_count} experiments")
        print("#"*80 + "\n")
        
        if not os.path.exists(config_dir):
            print(f"⚠️  Warning: Config directory not found: {config_dir}")
            print(f"   Skipping this group. Did you run generate_comprehensive_experiments.py?")
            failed_groups.append((name, "Config directory not found"))
            continue
        
        try:
            # Run batch experiments
            summary_df = run_batch_experiments(config_dir, "*.yaml")
            
            # Store results
            if summary_df is not None and len(summary_df) > 0:
                results_summary.append({
                    'group': name,
                    'expected': expected_count,
                    'completed': len(summary_df),
                    'status': 'SUCCESS' if len(summary_df) == expected_count else 'PARTIAL'
                })
                print(f"\n✓ {name} completed: {len(summary_df)}/{expected_count} experiments")
            else:
                results_summary.append({
                    'group': name,
                    'expected': expected_count,
                    'completed': 0,
                    'status': 'FAILED'
                })
                failed_groups.append((name, "No results generated"))
                print(f"\n❌ {name} failed: No results generated")
                
        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted by user")
            print(f"Stopping after group {i}/{len(experiments)}")
            break
            
        except Exception as e:
            print(f"\n❌ Error in {name}: {str(e)}")
            import traceback
            traceback.print_exc()
            failed_groups.append((name, str(e)))
            results_summary.append({
                'group': name,
                'expected': expected_count,
                'completed': 0,
                'status': 'ERROR'
            })
            
            # Ask if should continue
            response = input("\nContinue with remaining experiments? [y/N]: ")
            if response.lower() not in ['y', 'yes']:
                print("Stopping batch run.")
                break
    
    # Final summary
    end_time = datetime.now()
    duration = end_time - start_time
    
    print("\n" + "="*80)
    print("COMPREHENSIVE EXPERIMENTS COMPLETE")
    print("="*80)
    
    print(f"\nTotal time: {duration}")
    print(f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Ended: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    print("\nResults Summary:")
    print("-"*80)
    
    total_expected = 0
    total_completed = 0
    
    for result in results_summary:
        status_symbol = {
            'SUCCESS': '✓',
            'PARTIAL': '⚠',
            'FAILED': '❌',
            'ERROR': '❌'
        }.get(result['status'], '?')
        
        print(f"{status_symbol} {result['group']:<40} "
              f"{result['completed']:>3}/{result['expected']:<3} "
              f"({result['status']})")
        
        total_expected += result['expected']
        total_completed += result['completed']
    
    print("-"*80)
    print(f"{'TOTAL':<40} {total_completed:>3}/{total_expected:<3} "
          f"({total_completed/total_expected*100:.1f}%)")
    
    # Failed groups
    if failed_groups:
        print("\n  Failed/Incomplete Groups:")
        for name, reason in failed_groups:
            print(f"  - {name}: {reason}")
    
    # Next steps
    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print("\n1. Analyze results:")
    print("   python analyze_comprehensive_results.py")
    print("\n2. View results:")
    print("   cat results/analysis/summary_report.txt")
    print("   ls results/analysis/*.png")
    print("\n3. Check individual experiments:")
    print("   ls results/*/*/results.json")
    print("\n4. Load in Python:")
    print("   import pandas as pd")
    print("   df = pd.read_csv('results/all_results_combined.csv')")
    
    print("\n" + "="*80)
    
    # Return code based on success
    if total_completed == total_expected:
        print("\n All experiments completed successfully!")
        return 0
    elif total_completed > 0:
        print(f"\n  Partial completion: {total_completed}/{total_expected} experiments")
        return 1
    else:
        print("\n No experiments completed")
        return 2


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Exiting...")
        sys.exit(130)
    except Exception as e:
        print(f"\nFatal error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)