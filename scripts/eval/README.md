# Evaluation

We merge the weights of the last and best model checkpoints for evaluation. We provide the script `scripts/eval/merge_weights.py` which averages the weights of N model checkpoints. Or see the notebook `notebooks/merge_weights.ipynb`.

Usage

```bash 
python merge_weights.py --model_fnames "model1.ckpt,model2.ckpt" --output "merged.ckpt" --ckpt_dir "path/to/checkpoints"
```

## EnergyPlus ComStock 

For example, to evaluate the `medium/SSM_bert` model weights `checkpoints/ssm_medium_seed=1000_merged.pt` on the `short`, `medium`, and `long` captions

```bash
python scripts/eval/energyplus/eval.py --eval_name eval-comstock --resstock_comstock comstock --caption_splits short,medium,long --model medium/SSM_bert --ckpt_dir path/to/checkpoints --model_fnames ssm_medium_seed=1000_merged.pt`
```

## Wind Farm

For example, to evaluate the `keyvalue/ResNet` model weights `checkpoints/resnet_keyvalue_seed=1000_merged.pt` on the `keyvalue` captions


```bash
python scripts/eval/wind/eval.py --eval_name eval-wind --caption_splits keyvalue --model keyvalue/ResNet --ckpt_dir path/to/checkpoints --model_fnames resnet_keyvalue_seed=1000_merged.pt
```