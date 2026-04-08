#!/bin/bash
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
