import os 

for encoder_type in ['onehot', 'basic', 'medium']:
    for seed in [0, 1, 2]:
        os.system(f'''
                    python3 scripts/eval/merge_weights.py --model_fnames resnet_wind_{encoder_type}_seed={seed}_best.pt,resnet_wind_{encoder_type}_seed={seed}_last.pt --output resnet_wind_{encoder_type}_seed={seed}_merged.pt --ckpt_dir checkpoints/paper_renamed
                ''')