from transformers import DistilBertTokenizer, BertTokenizer, LongformerTokenizer
from syscaps.data.energyplus import EnergyPlusDataset
from syscaps import utils
from pathlib import Path
import os 
import torch
from tqdm import tqdm
import numpy as np
import argparse
import h5py 

## Env variables
SYSCAPS_PATH = os.environ.get('SYSCAPS', '')
if SYSCAPS_PATH == '':
    raise ValueError('SYSCAPS environment variable not set')
SYSCAPS_PATH = Path(SYSCAPS_PATH)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--simulator', type=str, default='energyplus_comstock',
                        help='energyplus or wind, default = energyplus')
    parser.add_argument('--index_files', type=str, default="all", 
                        help="index files, seperated by \",\", default = all")
    parser.add_argument('--caption_splits', type=str, default='keyvalue',
                        help='caption splits, separate by \",\", default = keyvalue')
    parser.add_argument('--tokenizer', type=str, default='distilbert-base-uncased',
                        help='tokenizer to use, default = distilbert-base-uncased')

    args = parser.parse_args()

    if args.index_files == 'all' and args.simulator == 'energyplus_comstock':
        args.index_files = ['comstock_train_seed=42.idx', 
                 'comstock_hyperparam_train_seed=42.idx', 
                 'comstock_buildings900k_test_seed=42.idx', 
                 'comstock_attribute_combos_seed=42.idx']
    elif args.index_files == 'all' and args.simulator == 'energyplus_resstock':
        args.index_files = ['resstock_train_seed=42.idx', 
                            'resstock_hyperparam_train_seed=42.idx', 
                            'resstock_buildings900k_test_seed=42.idx']
    elif args.index_files == 'all' and args.simulator == 'wind':
        args.index_files = ['floris_train_seed=42.idx', 'floris_val_seed=42.idx', 'floris_test_seed=42.idx']
    else:
        args.index_files = [idx_file.strip() for idx_file in args.index_files.split(",") if idx_file != ""]
    args.caption_splits = [caption_split.strip() for caption_split in args.caption_splits.split(",") if caption_split != ""]

    if args.tokenizer == 'distilbert-base-uncased':
        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    elif args.tokenizer == 'bert-base-uncased':
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    elif args.tokenizer == 'longformer-base-4096':
        tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')

    for idxf in args.index_files:
        for cs in args.caption_splits:
            
            print(f'idx file {idxf}, caption split {cs}')
            
            if 'energyplus' in args.simulator:
                if args.simulator == 'energyplus_comstock':
                    dataset = 'comstock'
                elif args.simulator == 'energyplus_resstock':
                    dataset = 'resstock'
            

                savedir = SYSCAPS_PATH / 'captions' / dataset
            
                savedir_ = savedir / f'{cs}_tokens' / args.tokenizer
                if not savedir_.exists():
                    os.makedirs(savedir_)

                dataset = EnergyPlusDataset(
                                data_path=Path(SYSCAPS_PATH),
                                index_file=idxf,
                                resstock_comstock=dataset,
                                syscaps_split=cs,
                                tokenizer = args.tokenizer,
                                return_full_year = True,
                                include_text = True
                )
                dataloader = torch.utils.data.DataLoader(
                    dataset,
                    batch_size=512,
                    drop_last=False,
                    num_workers = 16,
                    worker_init_fn=utils.worker_init_fn,
                    collate_fn=dataset.collate_fn())

                for batch in tqdm(dataloader):

                    building_ids = batch['building_id']
                    captions = batch['syscaps']

                    # remove part of caption before newline
                    stripped_caps = []
                    for c in captions:
                        if '\n' in c:
                            # find the first newline
                            c = c.split('\n')[1:]
                            # rejoin
                            c = '\n'.join(c)
                            stripped_caps += [c]
                        else:
                            stripped_caps += [c]

                    encoded_captions = tokenizer(stripped_caps, padding=False, truncation=False) 
                    #print(encoded_captions['input_ids'][0])
                    
                    #if idxf == 'comstock_train_seed=42.idx' and cs == 'long':
                    #    for ec in encoded_captions['input_ids']:
                    #        assert len(ec) > 2, ec

                    # save captions
                    for bldg_id,c in zip(building_ids,encoded_captions['input_ids']):
                        np.save( savedir_ / f'{bldg_id}_cap_ids.npy', c)
            
            elif args.simulator == 'wind':
                for style in [0,1,2,3]:
                    savedir = SYSCAPS_PATH / 'captions' / 'wind'
                    savedir_ = savedir / cs / f'aug_{style}_tokens' / args.tokenizer
                    if not savedir_.exists():
                        os.makedirs(savedir_)
                    data_path =  SYSCAPS_PATH / 'metadata' / 'syscaps' / \
                        'wind' / 'wind_plant_data.h5'

                    with h5py.File(data_path, 'r') as hf:
                        layout_names = [k for k in hf.keys() if 'Layout' in k]

                        captions = []
                        for ln in layout_names:
                            captions += [
                                open(SYSCAPS_PATH / 'captions' / 'wind' / cs / \
                                     f'aug_{style}' / f'{ln}_cap.txt').read()
                            ]

                        # remove part of caption before newline
                        stripped_caps = []
                        for c in captions:
                            if '\n' in c:
                                stripped_caps += [c.split('\n')[1]]
                            else:
                                stripped_caps += [c]

                        encoded_captions = tokenizer(stripped_caps, padding=False, truncation=False) 
                        # save captions
                        for l_id,c in zip(layout_names,encoded_captions['input_ids']):
                            np.save( savedir_ / f'{l_id}_cap_ids.npy', c)
