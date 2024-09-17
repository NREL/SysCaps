## Model training

See `scripts/train.sh`.

The script `scripts/train.py` is implemented with PyTorch `DistributedDataParallel` so it must be launched with `torchrun` from the command line and the argument `--disable_slurm` must be passed.


```bash
#!/bin/bash

export WORLD_SIZE=1
NUM_GPUS=1

torchrun \
    --nnodes=1 \
    --nproc_per_node=$NUM_GPUS \
    --rdzv-backend=c10d \
    --rdzv-endpoint=localhost:0 \
    scripts/train.py \
    --model LSTM --dataset energyplus_comstock --train_idx_file comstock_train_seed=42.idx --val_idx_file comstock_val_seed=42.idx --caption_dataset_split short --random_seed 1234 --disable_slurm
```

The argument `--disable_slurm` is not needed if you are running this script on a Slurm cluster as a batch job. 

This script will automatically log outputs to `wandb` if the environment variables `WANDB_ENTITY` and `WANDB_PROJECT` are set. Otherwise, pass the argument `--disable_wandb` to disable logging to `wandb`.


#### With SLURM

To launch training as a SLURM batch job:

```bash
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
echo "WORLD_SIZE="$WORLD_SIZE
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))

echo "NODELIST="${SLURM_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

srun python3 scripts/train.py \
--model LSTM --dataset energyplus_comstock --train_idx_file comstock_train_seed=42.idx --val_idx_file comstock_val_seed=42.idx --caption_dataset_split short --random_seed 1234 --disable_slurm
```

## Replicating the preprocessing steps

### Recursive Feature Elimination 

1. Create numpy datasets for RFE: # TODO
2. scripts/energyplus_feature_selection.py

### Index files

EnergyPlus Comstock:

```bash
python3 scripts/data_generation/create_energyplus_data_splits.py --resstock_comstock comstock --seed 1
```

### Data transforms

Fit the BoxCox transform using an index file:

```bash
python3 scripts/data_generation/fit_load_transform.py --energyplus_index_file comstock_hyperparam_train_seed=42.idx
```

This outputs the BoxCox pkl file to `$PROJECT/metadata/transforms/load`. Fit the weather transform using an index file:

```bash
python3 scripts/data_generation/fit_weather_transform.py --energyplus_index_file comstock_hyperparam_train_seed=42.idx
```
This outputs the StandardScaler params for each weather variable to `$PROJECT/metadata/transforms/weather/{col}`.


