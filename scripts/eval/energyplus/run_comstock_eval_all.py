import os 

for model in ['ResNet', 'LSTM', 'SSM']:
    for encoder_type in ['onehot', 'basic']:
        if model == 'SSM' and encoder_type == 'basic':
            continue 
        
        if model == 'ResNet' and encoder_type == 'basic':
            os.system(f'''
                        python3 scripts/energyplus/eval.py --eval_name aggregate_all --model_fnames {model.lower()}_{encoder_type}_seed=1000_best.pt,{model.lower()}_{encoder_type}_seed=1001_best.pt,{model.lower()}_{encoder_type}_seed=1002_best.pt --ckpt_dir checkpoints/paper_renamed --model {encoder_type}/{model} --caption_splits {encoder_type}
                    ''')
        else:
            
            os.system(f'''
                        python3 scripts/energyplus/eval.py --eval_name aggregate_all --model_fnames {model.lower()}_{encoder_type}_seed=1000_merged.pt,{model.lower()}_{encoder_type}_seed=1001_merged.pt,{model.lower()}_{encoder_type}_seed=1002_merged.pt --ckpt_dir checkpoints/paper_renamed --model {encoder_type}/{model} --caption_splits {encoder_type}
                    ''')

model = 'SSM'
encoder_type = 'medium'
os.system(f'''
            python3 scripts/energyplus/eval.py --eval_name aggregate_all --model_fnames {model.lower()}_{encoder_type}_seed=1000_merged.pt,{model.lower()}_{encoder_type}_seed=1001_merged.pt,{model.lower()}_{encoder_type}_seed=1002_merged.pt --ckpt_dir checkpoints/paper_renamed --model {encoder_type}/{model} --caption_splits {encoder_type}
        ''')