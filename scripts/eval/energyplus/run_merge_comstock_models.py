import os 

for model_type in ['resnet', 'lstm', 'ssm']:
    # TODO: medium
    for encoder_type in ['onehot', 'basic']:
        # TODO:
        if model_type == 'resnet' and encoder_type == 'basic':
            continue
        
        for seed in [1000, 1001, 1002]:
            os.system(f'''
                        python3 scripts/eval/merge_weights.py --model_fnames {model_type}_{encoder_type}_seed={seed}_best.pt,{model_type}_{encoder_type}_seed={seed}_last.pt --output {model_type}_{encoder_type}_seed={seed}_merged.pt --ckpt_dir checkpoints/paper_renamed
                    ''')