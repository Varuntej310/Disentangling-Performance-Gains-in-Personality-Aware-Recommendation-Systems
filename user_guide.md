# Complete Usage Guide: YAML-Based Experiments

## 🚀 Quick Start

### 1. Run a Single Experiment
```bash
python run_experiment.py config/experiments/baseline.yaml
```

### 2. Run Multiple Experiments (Batch)
```bash
python run_batch_experiments.py config/experiments/
```

### 3. Generate and Run Lambda Sweep
```bash
# Generate configs
python run_batch_experiments.py generate_lambda config/experiments/baseline.yaml

# Run all generated configs
python run_batch_experiments.py config/experiments/lambda_sweep/
```

---

## 📋 YAML Configuration Format

### Basic Structure
```yaml
experiment_name: "my_experiment"

paths:
  ratings: "path/to/ratings.csv"
  personality: "path/to/personality.csv"
  movies: "path/to/movies.csv"

data:
  personality_type: "real"  # real, shuffled, random, noise
  noise_distribution: null  # uniform, normal, laplace, bernoulli, exponential
  sparsity_percentile: 1.0
  min_interactions: 2

model:
  name: "Model_linear"  # Model_linear, Model_concat, Model_mtl, LightGCN
  hidden_channels: 5
  num_neighbors: [20, 10]
  personality_loss_weight: 0.0  # for MTL only

training:
  num_epochs: 50
  batch_size: 128
  lr: 0.001
  patience: 7
  num_runs: 3
  seed: 42
  device: "cuda"

eval:
  k_list: [3, 5, 10]
  num_negatives: 99

save_dir: "results/my_experiment"
```

---

## 🎯 Common Experiment Scenarios

### Scenario 1: Compare All Models

Create configs for each model:

**config/experiments/linear.yaml**
```yaml
experiment_name: "linear_model"
model:
  name: "Model_linear"
  hidden_channels: 5
training:
  num_runs: 5
save_dir: "results/comparison/linear"
```

**config/experiments/concat.yaml**
```yaml
experiment_name: "concat_model"
model:
  name: "Model_concat"
  hidden_channels: 5
training:
  num_runs: 5
save_dir: "results/comparison/concat"
```

**config/experiments/mtl.yaml**
```yaml
experiment_name: "mtl_model"
model:
  name: "Model_mtl"
  hidden_channels: 5
  personality_loss_weight: 0.5
training:
  num_runs: 5
save_dir: "results/comparison/mtl"
```

**Run all:**
```bash
python run_batch_experiments.py config/experiments/ "*model.yaml"
```

---

### Scenario 2: Hyperparameter Sweep (Lambda)

**Option A: Auto-generate configs**
```bash
# Generate configs for λ = 0.0, 0.25, 0.5, 0.75, 1.0
python run_batch_experiments.py generate_lambda config/experiments/mtl.yaml

# Run sweep
python run_batch_experiments.py config/experiments/lambda_sweep/
```

**Option B: Manual configs**

Create `config/experiments/lambda_0.0.yaml`, `lambda_0.5.yaml`, etc.:
```yaml
experiment_name: "mtl_lambda_0.5"
model:
  name: "Model_mtl"
  personality_loss_weight: 0.5
save_dir: "results/lambda_sweep/0.5"
```

---

### Scenario 3: Test Different Personality Types

**Real personality:**
```yaml
experiment_name: "real_personality"
data:
  personality_type: "real"
save_dir: "results/ablation/real"
```

**Shuffled personality:**
```yaml
experiment_name: "shuffled_personality"
data:
  personality_type: "shuffled"
save_dir: "results/ablation/shuffled"
```

**Random uniform:**
```yaml
experiment_name: "random_personality"
data:
  personality_type: "random"
save_dir: "results/ablation/random"
```

**Noise distributions:**
```yaml
experiment_name: "noise_normal"
data:
  personality_type: "noise"
  noise_distribution: "normal"
save_dir: "results/ablation/noise_normal"
```

---

### Scenario 4: Learning Rate Sweep

Create configs with different learning rates:

```yaml
# config/experiments/lr_0.001.yaml
training:
  lr: 0.001
save_dir: "results/lr_sweep/0.001"

# config/experiments/lr_0.01.yaml
training:
  lr: 0.01
save_dir: "results/lr_sweep/0.01"

# config/experiments/lr_0.0001.yaml
training:
  lr: 0.0001
save_dir: "results/lr_sweep/0.0001"
```

---

### Scenario 5: Quick Testing (Small Run)

For rapid iteration:
```yaml
experiment_name: "quick_test"
training:
  num_epochs: 10
  patience: 3
  num_runs: 1
  batch_size: 256  # larger for speed
```

---

## 📊 Understanding Results

### Output Structure
```
results/
├── my_experiment/
│   ├── config.yaml          # Copy of experiment config
│   └── results.json         # Detailed results
└── batch_summary_20240115.csv  # Batch run summary
```

### Results JSON Format
```json
{
  "experiment_name": "baseline_linear",
  "model": "Model_linear",
  "num_runs": 3,
  "avg_metrics": {
    "HR@3": 0.1234,
    "HR@5": 0.2345,
    "HR@10": 0.3456,
    "NDCG@3": 0.0987,
    "NDCG@5": 0.1234,
    "NDCG@10": 0.1567
  },
  "std_metrics": {
    "HR@3_std": 0.0123,
    ...
  },
  "all_runs": [
    {"run": 1, "HR@10": 0.3401, ...},
    {"run": 2, "HR@10": 0.3512, ...},
    {"run": 3, "HR@10": 0.3455, ...}
  ],
  "config": {...},
  "timestamp": "2024-01-15T10:30:00"
}
```

---

## 🔧 Advanced Usage

### Custom Python Script
```python
from config.experiment_config import ExperimentConfig
from run_experiment import run_single_experiment

# Load and modify config
config = ExperimentConfig.from_yaml('config/experiments/baseline.yaml')
config.training.num_epochs = 100
config.model.hidden_channels = 10
config.save_dir = 'results/custom'

# Save modified config
config.to_yaml('config/experiments/custom.yaml')

# Run experiment
run_single_experiment('config/experiments/custom.yaml')
```

### Programmatic Config Generation
```python
from config.experiment_config import ExperimentConfig, ModelConfig, TrainingConfig

# Create config programmatically
config = ExperimentConfig(
    experiment_name="programmatic_exp",
    model=ModelConfig(
        name="Model_mtl",
        hidden_channels=8,
        personality_loss_weight=0.6
    ),
    training=TrainingConfig(
        num_epochs=30,
        lr=0.002,
        num_runs=5
    ),
    save_dir="results/programmatic"
)

# Save and run
config.to_yaml('config/experiments/prog.yaml')
run_single_experiment('config/experiments/prog.yaml')
```

### Batch Processing with Custom Logic
```python
import glob
from run_experiment import run_single_experiment

# Get all configs
configs = glob.glob('config/experiments/mtl_*.yaml')

# Run only specific ones
for cfg in configs:
    if 'lambda' in cfg:
        print(f"Running {cfg}...")
        run_single_experiment(cfg)
```

---

## 📈 Analysis and Visualization

### Load and Analyze Results
```python
import json
import pandas as pd
import matplotlib.pyplot as plt

# Load single experiment
with open('results/my_exp/results.json') as f:
    results = json.load(f)

print(f"NDCG@10: {results['avg_metrics']['NDCG@10']:.4f}")

# Load batch summary
summary = pd.read_csv('results/batch_summary_20240115.csv')

# Plot comparison
summary.plot(x='experiment', y='NDCG@10', kind='bar')
plt.tight_layout()
plt.savefig('results/comparison.png')
```

### Lambda Sweep Visualization
```python
import pandas as pd
import matplotlib.pyplot as plt

# Load lambda sweep results
lambda_results = []
for lam in [0.0, 0.25, 0.5, 0.75, 1.0]:
    with open(f'results/lambda_sweep/lambda_{lam}/results.json') as f:
        data = json.load(f)
        lambda_results.append({
            'lambda': lam,
            'NDCG@10': data['avg_metrics']['NDCG@10'],
            'std': data['std_metrics']['NDCG@10_std']
        })

df = pd.DataFrame(lambda_results)

# Plot with error bars
plt.figure(figsize=(10, 6))
plt.errorbar(df['lambda'], df['NDCG@10'], yerr=df['std'], 
             marker='o', capsize=5)
plt.xlabel('Personality Loss Weight (λ)')
plt.ylabel('NDCG@10')
plt.title('MTL Performance vs Lambda')
plt.grid(True, alpha=0.3)
plt.savefig('results/lambda_sweep.png', dpi=300, bbox_inches='tight')
```

---

## ⚡ Tips and Best Practices

1. **Start Small**: Test with `num_runs=1` and `num_epochs=10` first
2. **Use Descriptive Names**: Name experiments clearly (e.g., `mtl_lambda0.5_lr0.001`)
3. **Version Control**: Commit YAML configs to git
4. **Batch Processing**: Use batch runner for systematic sweeps
5. **Save Everything**: Results include full config for reproducibility
6. **Monitor GPU**: Use `nvidia-smi` to check memory usage
7. **Parallel Runs**: Manually run different configs on different GPUs:
   ```bash
   CUDA_VISIBLE_DEVICES=0 python run_experiment.py config1.yaml &
   CUDA_VISIBLE_DEVICES=1 python run_experiment.py config2.yaml &
   ```

---

## 🐛 Troubleshooting

### Out of Memory
```yaml
training:
  batch_size: 64  # reduce from 128
model:
  num_neighbors: [10, 5]  # reduce from [20, 10]
```

### Too Slow
```yaml
training:
  num_epochs: 20  # reduce from 50
  patience: 3     # reduce from 7
  num_runs: 1     # reduce from 3
```

### Results Not Improving
```yaml
training:
  lr: 0.01        # try higher learning rate
  num_epochs: 100 # try more epochs
  patience: 10    # increase patience
```

---

## 📚 Complete Workflow Example

```bash
# 1. Create base config
cat > config/experiments/base.yaml << EOF
experiment_name: "base"
model:
  name: "Model_mtl"
  hidden_channels: 5
training:
  num_epochs: 50
  num_runs: 3
save_dir: "results/base"
EOF

# 2. Generate lambda sweep
python run_batch_experiments.py generate_lambda config/experiments/base.yaml

# 3. Run sweep
python run_batch_experiments.py config/experiments/lambda_sweep/

# 4. Analyze results
python -c "
import pandas as pd
summary = pd.read_csv('results/batch_summary_*.csv')
print(summary[['experiment', 'NDCG@10']].sort_values('NDCG@10', ascending=False))
"
```

This approach makes experimentation systematic, reproducible, and easy to manage!