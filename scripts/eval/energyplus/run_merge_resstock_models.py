import os 

for encoder_type in ['onehot', 'basic', 'long']:
    for seed in [2000, 2001, 2002]:
        os.system(f'''
                    python3 scripts/eval/merge_weights.py --model_fnames ssm_resstock_{encoder_type}_seed={seed}_best.pt,ssm_resstock_{encoder_type}_seed={seed}_last.pt --output ssm_resstock_{encoder_type}_seed={seed}_merged.pt --ckpt_dir checkpoints/paper_renamed
                ''')