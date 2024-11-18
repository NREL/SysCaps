## Grid search hyperparameter sweep 

Create a TOML file called `{caption_split}/{model_name}_hyperopt.toml` in the `configs` directory. Use a list to specify the hyperparameters to perform a grid search over. For example, to perform a grid search over the learning rate and batch size for the `medium/SSM_bert` model, create the following file `configs/energyplus_comstock/medium/SSM_bert_hyperopt.toml` with only these lines:

```toml
[experiment]
lr = [0.001, 0.01, 0.1]
batch_size = [16, 32, 64]
```

Then provide this file as an argument to `scripts/train.py`: 

```bash
torchrun \
    --nnodes=1 \
    --nproc_per_node=1 \
    --rdzv-backend=c10d \
    --rdzv-endpoint=localhost:0 \
    scripts/train.py \
        --model medium/SSM_bert \ 
        --hyperopt_file medium/SSM_bert_hyperopt.toml \
        --dataset energyplus_comstock \ 
        --train_idx_file comstock_train_seed=42.idx \ 
        --val_idx_file comstock_val_seed=42.idx \
        --caption_dataset_split medium \  
         --random_seed 1234 \
         --disable_slurm
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

This outputs the BoxCox pkl file to `<SYSCAPS>/metadata/transforms/load`. Fit the weather transform using an index file:

```bash
python3 scripts/data_generation/fit_weather_transform.py --energyplus_index_file comstock_hyperparam_train_seed=42.idx
```
This outputs the StandardScaler params for each weather variable to `$PROJECT/metadata/transforms/weather/{col}`.


