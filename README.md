# Semantic Enhancement or Regularization?

### Disentangling Performance Gains in Personality-Aware Recommendation Systems

Official implementation for:

**Semantic Enhancement or Regularization? Disentangling Performance Gains in Personality-Aware Recommendation Systems**
Abhradeep Datta*, Varun Tej Kasula*, Ashok Singh Sairam
Indian Institute of Technology Guwahati
(* equal contribution)

## Setup

Create a Python environment:

```bash
conda create -n gpu python=3.10 -y
conda activate gpu
```

Install PyTorch (CUDA 12.1):

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Install PyTorch Geometric dependencies:

```bash
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv \
-f https://data.pyg.org/whl/torch-2.5.1+cu121.html
```

Install remaining requirements:

```bash
pip install -r requirements.txt
```

### Older Systems

If installation fails with:

```text
OSError: /lib64/libm.so.6: version `GLIBC_2.29' not found
```

remove `pyg-lib`, which requires GLIBC >= 2.29.

## Running Experiments

Generate experiment configurations:

```bash
python -m experiments.generate_ablation_experiments
```

Run ablation experiments:

```bash
python -m experiments.run_ablation_experiments config/experiments/ablation_table/
```

## Example Reproduction

The following commands reproduce the ablation study results for personality 2018 dataset

```bash
python -m experiments.generate_ablation_experiments
python -m experiments.run_ablation_experiments config/experiments/ablation_table/
python -m experiments.format_ablation_latex
```

## Running on a GPU Server

Start a persistent tmux session:

```bash
tmux new -s work
```

Inside the session:

```bash
conda activate gpu
export CUDA_VISIBLE_DEVICES=0
```

Run experiments normally:

```bash
python -m experiments.run_ablation_experiments config/experiments/ablation_table/
```

Detach without stopping the job:

```text
Ctrl+B, then D
```

Reconnect later:

```bash
tmux attach -t work
```

## Data

Datasets are not distributed with this repository.
