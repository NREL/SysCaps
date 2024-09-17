import os 

for encoder_type in ['onehot', 'basic', 'long']:

    os.system(f'''
                python3 scripts/eval/energyplus/eval.py --eval_name eval-resstock --model_fnames ssm_resstock_{encoder_type}_seed=2000_merged.pt,ssm_resstock_{encoder_type}_seed=2001_merged.pt,ssm_resstock_{encoder_type}_seed=2002_merged.pt --ckpt_dir checkpoints/paper_renamed --model {encoder_type}/SSM --caption_splits {encoder_type} --resstock_comstock resstock
            ''')

