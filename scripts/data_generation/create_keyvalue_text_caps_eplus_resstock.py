from transformers import DistilBertTokenizer, BertTokenizer, LongformerTokenizer
from syscaps.data.energyplus import EnergyPlusDataset
from syscaps import utils
from pathlib import Path
import os 
import torch
from tqdm import tqdm
import numpy as np

if __name__ == '__main__':
    tok_type = "longformer-base-4096"
    if tok_type == "bert-base-uncased":
        tokenizer = BertTokenizer.from_pretrained(tok_type)
    elif tok_type == 'longformer-base-4096':
        tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
    else:
        tokenizer = DistilBertTokenizer.from_pretrained(tok_type)

    ## Env variables
    SYSCAPS_PATH = os.environ.get('SYSCAPS', '')
    if SYSCAPS_PATH == '':
        raise ValueError('SYSCAPS_PATH environment variable not set')
    SYSCAPS_PATH = Path(SYSCAPS_PATH)
    
    attributes = open(SYSCAPS_PATH / 'metadata' / 'attributes_resstock.txt', 'r').read().split('\n')
    attributes = [x.strip('"') for x in attributes]
    attributes = [x for x in attributes if x != '']
    # strip in.
    attributes = [x.replace('in.','') for x in attributes]
    # strip _
    attributes = [x.replace('_',' ') for x in attributes]

    savedir = SYSCAPS_PATH / 'captions' / 'resstock'

    idx_files = ['resstock_train_seed=42.idx', 
                 'resstock_hyperparam_train_seed=42.idx', 
                 'resstock_buildings900k_test_seed=42.idx']

    for idxf in idx_files:
            
        savedir_ = savedir / f'basic'
        savedir_tokens = savedir / f'basic_tokens' / f'{tok_type}'

        if not savedir_.exists():
            os.makedirs(savedir_)
        if not savedir_tokens.exists():
            os.makedirs(savedir_tokens)

        dataset = EnergyPlusDataset(
                        buildings_bench_path=SYSCAPS_PATH,
                        index_file=idxf,
                        resstock_comstock='resstock',
                        syscaps_split='long',
                        tokenizer=tok_type,
                        return_full_year = True,
                        include_text = True
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=512,
            drop_last=False,
            num_workers = 8,
            worker_init_fn=utils.worker_init_fn,
            collate_fn=dataset.collate_fn())

        for batch in tqdm(dataloader):

            building_ids = batch['building_id']
            onehot = batch['attributes_onehot']
            onehot_np = onehot.numpy()
            attrs = dataset.attribute_onehot_encoder.inverse_transform(onehot_np[:,0])
            #for batch_idx in range(onehot.shape[0]):
            #    attrs = dataset.attribute_onehot_encoder.inverse_transform(
            #        onehot_np[batch_idx]
            #    )
            captions = []
            for j in range(attrs.shape[0]):
                attr_str = ''
                for i in range(len(attributes)):
                   
                    attr_str += f'{attributes[i]}:{attrs[j,i]}|'
                captions += [attr_str[:-1]]

            encoded_captions = tokenizer(captions, padding=False, truncation=False) 

            for bldg_id,c in zip(building_ids, captions):
                with open( savedir_ / f'{bldg_id}_cap.txt', 'w') as f:
                    f.write(c)
            # save tokenized_captions
            for bldg_id,c in zip(building_ids,encoded_captions['input_ids']):
                np.save( savedir_tokens / f'{bldg_id}_cap_ids.npy', c)