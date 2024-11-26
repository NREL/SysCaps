# Welcome to SysCaps

You've found the official code repository for the paper [SysCaps: Language Interfaces for Simulation Surrogates of Complex Systems](https://arxiv.org/abs/2405.19653), which will be presented at the upcoming Foundation Models for Science: Progress, Opportunities, and Challenges workshop at [NeurIPS 2024](https://fm-science.github.io/). Our paper conjectures that interfaces (both text templates as well as conversational) makes interacting with simulation surrogate models for complex systems more intuitive and accessible for both non-experts and experts. "System captions", or SysCaps, are text-based descriptions of systems based on information contained in simulation metadata. Our paper's goal is to train multimodal regression models that take text inputs (SysCaps) and timeseries inputs (exogenous system conditions such as hourly weather) and regress timeseries simulation outputs (e.g. hourly building energy consumption). The experiments in our paper with building and wind farm simulators, which can be reproduced using this codebase, aim to help us understand whether a) accurate regression in this setting is possible and b) if so, how well can we do it. 

## Getting started

Clone this repository and install it in editable mode in a conda environment.

1. Create an environment with `python>=3.9`, for example: `conda create -n syscaps python=3.9`.
2. Install the repo in editable mode with

```bash
git clone git@github.com:NREL/SysCaps.git
cd SysCaps
pip install -e .
```

You should have an NVIDIA GPU available to do the install and run the code.

### Install CUDA kernels for the SSM

See https://github.com/state-spaces/s4/tree/main#structured-kernels. Ensure your `$CUDA_HOME` environment variable is properly set and gcc is available and up to date. On NREL Kestrel, this is `/nopt/cuda/12.3`. You may need to run `module load gcc` first.

Navigate to `./syscaps/models/third_party/extensions/kernels` and run the following command:

```bash
python setup.py install
```

### Downloading all data

If you are starting from scratch, we recommend using the following script to download all datasets and metadata for this repository to a folder `DATA_DIR`:

```bash
bash prepare_data.sh <DATA_DIR>
```

The script will download and untar the BuildingsBench datasets (see the [BuildingsBench docs](https://nrel.github.io/BuildingsBench/getting_started/) for more info about this) into `DATA_DIR`, as well as the SysCaps-specific data and metadata from [HuggingFace](https://huggingface.co/datasets/NREL/SysCaps). This uses `huggingface_hub`, so you'll have to have this installed in your Python environment first. All of the data and metadata will be stored under `$DATA_DIR/BuildingsBench`:

Folder organization:

```bash
# building weather, simulation metadata, and load timeseries
$DATA_DIR/BuildingsBench/Buildings-900K/...
# SysCaps buildings and wind captions data 
$DATA_DIR/BuildingsBench/captions
# SysCaps-specific metadata files, including wind firm data
$DATA_DIR/BuildingsBench/metadata/syscaps
# Other BuildingsBench folders, unused...
# ...
```

#### Tokenization 

Use provided script `scripts/tokenize_captions.py` to tokenize the captions data after downloading it. 

Example: `python scripts/tokenize_captions.py --simulator energyplus_comstock --caption_splits keyvalue,short,medium,long --tokenizer bert-base-uncased`

### Environment Variables

* `SYSCAPS`: The `prepare_data.sh` script sets this environment variable to `$DATA_DIR/BuildingsBench` for you. If not using that script, make sure to set it to the path of the `BuildingsBench` folder where you stored the data.

Other environment variables that are used in the codebase:

* `WANDB_ENTITY`=YOUR WANDB ENTITY NAME
* `WANDB_PROJECT`=YOUR WANDB PROJECT NAME

## Quickstart with captions data and pretrained models

You can check out the LLM-generated and key-value template text captions using HuggingFace datasets without much effort:

```python
from datasets import load_dataset

# configs: 'comstock', 'wind'
dataset = load_dataset('NREL/SysCaps', 'comstock')

print(dataset['medium']['caption'][0])
```

You can also load a selection of pretrained models from HuggingFace for evaluation as follows:

```python
# We use TOML config files to save model and experiment arguments
all_args = tomli.load( Path( 'syscaps/configs/energyplus_comstock/keyvalue/SSM_bert.toml' ).open('rb') )
# Use `model_factory` to load the model from the config
model = model_factory(all_args['experiment']['module_name'], 'energyplus_comstock', all_args['model'])
# Load the pretrained model weights from HuggingFace
model.from_pretrained('NREL/building-surrogate-kv')
```

* `NREL/building-surrogate-kv`, config: `energyplus_comstock/keyvalue/SSM_bert.toml`
* `NREL/building-surrogate-nl`, config: `energyplus_comstock/medium/SSM_bert.toml`
* `NREL/building-surrogate-onehot`, config: `energyplus_comstock/onehot/SSM.toml`
* `NREL/wind-farm-surrogate-kv`, config: `wind/keyvalue/ResNet.toml`
* `NREL/wind-farm-surrogate-nl`, config: `wind/medium/ResNet.toml`
* `NREL/wind-farm-surrogate-onehot`, config: `wind/onehot/ResNet.toml`

## Model training

Train surrogate models with `scripts/train.py`. 
We use PyTorch `DistributedDataParallel`, so launch it with `torchrun` from the command line.  Use the argument `--disable_slurm` if not training with SLURM.

```bash
torchrun \
    --nnodes=1 \
    --nproc_per_node=1 \
    --rdzv-backend=c10d \
    --rdzv-endpoint=localhost:0 \
    scripts/train.py \
        --model medium/SSM_bert \  # name of the config
        --dataset energyplus_comstock \ # energyplus_comstock or wind
        --train_idx_file comstock_train_seed=42.idx \ # index files dataloader. In metadata/syscaps/splits
        --val_idx_file comstock_val_seed=42.idx \
        --caption_dataset_split medium \  # energyplus_comstock: short, medium, long, keyvalue
         --random_seed 1234 \
         --disable_slurm
```

See `scripts/train.py` for the full list of command line arguments.
This script will automatically log outputs to `wandb` if the environment variables `WANDB_ENTITY` and `WANDB_PROJECT` are set. Otherwise, pass the argument `--disable_wandb` to disable logging to `wandb`.

For more training and data preprocessing details, see the README.md in the `scripts` directory.

## Model evals

See README.md in `scripts/eval` folder.

### Generating the SysCaps with Llama-2

To re-create the SysCaps data, follow instructions in the README.md in the `llama` directory.

## Evaluating caption quality using attribute classifiers

See `scripts/training_caption_attribute_classifier.py`.


## In development

- [ ] Add a notebook for exploring the pretrained model weights
- [ ] Add HuggingFace models to evaluation scripts

## Citation

If you use this code in your research, please cite the following paper:

```
@article{emami2024syscaps,
  title={SysCaps: Language Interfaces for Simulation Surrogates of Complex Systems},
  author={Emami, Patrick and Li, Zhaonan and Sinha, Saumya and Nguyen, Truc},
  journal={arXiv preprint arXiv:2405.19653},
  year={2024}
}
```