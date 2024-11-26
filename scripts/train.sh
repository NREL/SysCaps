#!/bin/bash

export WORLD_SIZE=1
NUM_GPUS=1
export WANDB_PROJECT=test

CUDA_VISIBLE_DEVICES=0 torchrun \
    --nnodes=1 \
    --nproc_per_node=$NUM_GPUS \
    --rdzv-backend=c10d \
    --rdzv-endpoint=localhost:0 \
    scripts/train.py \
        --model onehot/LSTM \
        --note lstm_test_comstock_seed=10 \
        --dataset energyplus_comstock \
        --train_idx_file comstock_train_seed=42.idx \
        --val_idx_file comstock_val_seed=42.idx \
        --random_seed 10 \
        --num_workers 8 \
        --disable_slurm \
        --disable_wandb
