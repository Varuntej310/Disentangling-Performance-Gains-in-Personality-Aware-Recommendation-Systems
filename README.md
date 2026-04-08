# Semantic Enhancement or Regularization?
### Disentangling Performance Gains in Personality-Aware Recommendation Systems

This repository contains the official implementation for the paper:

> **Semantic Enhancement or Regularization? Disentangling Performance Gains in Personality-Aware Recommendation Systems**  
> Abhradeep Datta, Varun Tej Kasula, Ashok Singh Sairam  
> IIT Guwahati

---

# Disentangling Performance Gains in Personality-Aware Recommendation Systems

This repository contains the implementation of experiments analyzing how personality features influence graph-based recommendation systems. The focus is on understanding performance gains through factors such as sparsity, noise injection, and model design.

---

## Overview

This project investigates:

* The role of personality features in recommendation systems
* The effect of graph sparsity on model performance
* The impact of noise injection strategies
* Comparisons between different model architectures (e.g., LightGCN, GAT, NGCF)
* Multi-task learning vs. simple feature concatenation

---

## Requirements

* Python 3.10+
* PyTorch 2.5.1 (CUDA 12.1 recommended)
* PyTorch Geometric dependencies
* Additional packages listed in `requirements.txt`

---

## Installation

```bash
# Create environment
conda create -n gpu python=3.10 -y
conda activate gpu

# Install PyTorch (CUDA 12.1)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install PyTorch Geometric dependencies
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv \
-f https://data.pyg.org/whl/torch-2.5.1+cu121.html

# Install remaining dependencies
pip install -r requirements.txt
```

---

## Project Structure

```
├── config/                  # Configuration files
├── data/                    # Dataset loading modules
├── models/                  # Model implementations
├── training/                # Training and evaluation logic
├── experiments/             # Experiment scripts
├── utils/                   # Helper functions
├── notebooks/               # Jupyter notebooks
├── requirements.txt
└── README.md
```

---

## Datasets

This project uses:

### 1. MovieLens (Enhanced with Personality Features)

* User-item interaction data
* Personality feature vectors

### 2. LastFM Dataset

* User-artist interaction data
* Configurations available in `config/experiments_lastfm/`

**Note:** Datasets are not included in this repository.

---

## Running Experiments

All experiments are configuration-driven.

### Example Workflows

```bash
# Generate ablation configs
python -m experiments.generate_ablation_experiments

# Run ablation experiments
python -m experiments.run_ablation_experiments config/experiments/ablation_table/

# Run batch experiments
python -m experiments.run_batch_experiments config/experiments/

# Run LastFM experiments
python -m experiments.run_lastfm_experiments config/experiments_lastfm/
```

---

## GPU Execution (Recommended)

```bash
tmux new -s experiments

conda activate gpu
export CUDA_VISIBLE_DEVICES=0

python -m experiments.run_batch_experiments config/experiments/

# Detach: Ctrl+B → D
# Reattach: tmux attach -t experiments
```

---

## Experiment Configuration

Experiments are defined using YAML files:

```yaml
name: "experiment_name"
model: "lightgcn"
dataset: "base"
personality_injection: "add"
noise_type: "gaussian"
noise_magnitude: 0.1
sparsity_level: 0.05
lambda_param: 0.5
```

---

## Key Insights

* Graph sparsity significantly affects recommendation quality
* Noise injection can destabilize training depending on type and magnitude
* Multi-task learning improves robustness over simple concatenation
* Different models respond differently to personality features

