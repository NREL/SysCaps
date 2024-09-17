# SysCaps: System Captions 

Clone this repository and install it in editable mode in a conda environment.

1. Create an environment with `python>=3.8`, for example: `conda create -n syscaps python=3.9`.
2. Install the repo in editable mode with

```bash
git clone git@github.com:NREL/SysCaps.git
cd SysCaps
pip install -e .
```

You should have an NVIDIA GPU available to do the install and run the code.

#### Dependencies for S4

See https://github.com/state-spaces/s4/tree/main#structured-kernels. Ensure your $CUDA_HOME environment variable is properly set and gcc is available and up to date. On Kestrel, $CUDA_HOME is `/nopt/cuda/12.3` and you may need to run `module load gcc`.

Navigate to syscaps/models/third_party/extensions/kernels and run the following command:

```bash
python setup.py install
```

### Data

Follow the instructions to download the BuildingsBench dataset `Buildings-900K` provided [here](https://github.com/NREL/BuildingsBench/?tab=readme-ov-file#download-the-datasets-and-metadata).

Follow the instructions to download the SysCaps captions data provided [here](). We will store all data for this project under the `BuildingsBench` folder.

Assuming `$DATA_DIR` is the directory where you saved the data, the data should be organized as follows:

```bash
# building weather and load timeseries
$DATA_DIR/BuildingsBench/Buildings-900K/...
# SysCaps captions data and wind datasest data
$DATA_DIR/BuildingsBench/captions
```

The SysCaps metadata will be saved in the directory `$DATA_DIR/BuildingsBench/metadata`. The metadata includes the normalization parameters for the model inputs and outputs, as well as the index files indicating the train, val, and test splits. 


### Environment variables

```bash
export SYSCAPS=$DATA_DIR/BuildingsBench
export WANDB_ENTITY=<YOUR WANDB ENTITY NAME>
export WANDB_PROJECT=<YOUR WANDB PROJECT NAME>
```

## Getting Started

TODO: Add instructions for downloading trained models from HF, running evals, plotting results, etc.

## Generating System Captions with Llama-2

TODO: See `llama/`. 

## Evaluating caption quality using attribute classifiers

TODO: See `scripts/training_caption_attribute_classifier.py`.

## Surrogate model training

See `scrips/train.sh`.

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

### Notes on ComStock building attributes

RFE excluded attributes:

- in.rotation
- in.heating_fuel
- in.number_stories (there's another attribute with number storeys that is kept)
- in.service_water_heating_fuel
- in.aspect_ratio

Attributes per attribute type = [10, 14, 16, 10, 64, 53, 36, 53, 36, 10, 12, 10, 12]