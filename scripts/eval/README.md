## caption splits

#### Comstock

- keyvalue
- short
- medium
- long
- medium_missing_building_type
- medium_building_type_warehouse

## models

### ComStock

- basic/LSTM: `lstm_basic_seed=1000_merged.pt`, `lstm_basic_seed=1001_merged.pt`, `lstm_basic_seed=1002_merged.pt`
- onehot/LSTM: `lstm_onehot_seed=1000_merged.pt`, `lstm_onehot_seed=1001_merged.pt`, `lstm_onehot_seed=1002_merged.pt`
- basic/ResNet: `resnet_basic_seed=1000_best.pt`, `resnet_basic_seed=1001_best.pt`, `resnet_basic_seed=1002_best.pt`
- onehot/ResNet: `resnet_onehot_seed=1000_merged.pt`, `resnet_onehot_seed=1001_merged.pt`, `resnet_onehot_seed=1002_merged.pt`
- medium/SSM: `ssm_medium_seed=1000_merged.pt`, `ssm_medium_seed=1001_merged.pt`, `ssm_medium_seed=1002_merged.pt`
- basic/SSM: `ssm_basic_seed=1000_merged.pt`, `ssm_basic_seed=1001_merged.pt`, `ssm_basic_seed=1002_merged.pt`
- onehot/SSM: `ssm_onehot_seed=1000_merged.pt`, `ssm_onehot_seed=1001_merged.pt`, `ssm_onehot_seed=1002_merged.pt`
- medium/SSM_bert: `ssm_bert_medium_seed=1000_merged.pt`, `ssm_bert_medium_seed=1001_merged.pt`, `ssm_bert_medium_seed=1002_merged.pt`

### ResStock

### Wind

- onehot/ResNet: `resnet_wind_onehot_seed=0_merged.pt`, `resnet_wind_onehot_seed=1_merged.pt`, `resnet_wind_onehot_seed=2_merged.pt`
- basic/ResNet: `resnet_wind_basic_seed=0_merged.pt`, `resnet_wind_basic_seed=1_merged.pt`, `resnet_wind_basic_seed=2_merged.pt`
- medium/ResNet: `resnet_wind_medium_seed=0_merged.pt`, `resnet_wind_medium_seed=1_merged.pt`, `resnet_wind_medium_seed=2_merged.pt`

## experiments

#### NRMSE computed over all buildings 

- evaluate on onehot/basic/short/medium/long-length test captions, four aggregation levels
    - `python3 scripts/eval/energyplus/eval.py --eval_name eval-comstock --caption_splits {onehot/basic/short/medium/long} --model {onehot/basic/medium}/SSM --model_fnames ssm_medium_seed=1000_merged.pt,ssm_medium_seed=1001_merged.pt,ssm_medium_seed=1002_merged.pt ...`
- evaluate on medium-length test captions with missing building type+sub_type, four aggregation levels
    - `python3 scripts/eval/energyplus/eval.py --eval_name eval-comstock --caption_splits medium_missing_building_type --model medium/SSM --model_fnames ssm_medium_seed=1000_merged.pt,ssm_medium_seed=1001_merged.pt,ssm_medium_seed=1002_merged.pt ...`


#### NRMSE computed per building type

- evaluate on OOD building types, per building hourly aggregation level 
    - `python3 scripts/eval/energyplus/eval_comstock_building_type_gen.py --caption_splits {medium_building_type_motel, medium_building_type_bb, medium_building_type_warehouse} --model medium/SSM --model_fnames ssm_medium_seed=1000_merged.pt,ssm_medium_seed=1001_merged.pt,ssm_medium_seed=1002_merged.pt ...`
- evaluate on medium-length test captions with missing building type+sub_type, per-building-type performance, per building hourly aggregation level
    - `python3 scripts/eval/energyplus/eval_comstock_building_type_gen.py --caption_splits medium_missing_building_type --model medium/SSM --model_fnames ssm_medium_seed=1000_merged.pt,ssm_medium_seed=1001_merged.pt,ssm_medium_seed=1002_merged.pt  ...`
    
#### per-attribute results 

- evaluate on random missing attributes, per-attribute performance, per building hourly aggregation level 
    - `python3 scripts/eval/energyplus/eval_comstock_missing.py --caption_splits medium,medium_missing --model medium/SSM --model_fnames ssm_medium_seed=1000_merged.pt,ssm_medium_seed=1001_merged.pt,ssm_medium_seed=1002_merged.pt ...`

#### qualitative missing attributes figure

