from transformers import DistilBertTokenizer, BertTokenizer
from syscaps.data.energyplus import EnergyPlusDataset
from syscaps import utils
from pathlib import Path
import os 
import torch
from tqdm import tqdm
import numpy as np

if __name__ == '__main__':
    tok_type = "bert-base-uncased"
    if tok_type == "bert-base-uncased":
        tokenizer = BertTokenizer.from_pretrained(tok_type)
    else:
        tokenizer = DistilBertTokenizer.from_pretrained(tok_type)

    ## Env variables
    SYSCAPS_PATH = os.environ.get('SYSCAPS', '')
    if SYSCAPS_PATH == '':
        raise ValueError('SYSCAPS_PATH environment variable not set')
    SYSCAPS_PATH = Path(SYSCAPS_PATH)

    attributes = open(SYSCAPS_PATH / 'metadata' / 'attributes_comstock.txt', 'r').read().split('\n')
    # strip ""
    attributes = [x.strip('"') for x in attributes]
    # strip in.
    attributes = [x.replace('in.','') for x in attributes]
    # strip _
    attributes = [x.replace('_',' ') for x in attributes]

    building_types = {
        'FullServiceRestaurant': 'full service restaurant',
        'RetailStripmall': 'retail strip mall',
        'Warehouse': 'warehouse',
        'RetailStandalone': 'retail standalone',
        'SmallOffice': 'small office',
        'PrimarySchool': 'primary school',
        'MediumOffice': 'medium office',
        'SecondarySchool': 'secondary school',
        'Outpatient': 'outpatient',
        'QuickServiceRestaurant': 'quick service restaurant',
        'LargeOffice': 'large office',
        'LargeHotel': 'large hotel',
        'SmallHotel': 'small hotel',
        'Hospital': 'hospital'
    }
    building_subtypes  = {
        'strip_mall_restaurant20': 'strip mall restaurant20',
        'mediumoffice_nodatacenter': 'medium office no datacenter',
        'strip_mall_restaurant30':'strip mall restaurant20',
        'strip_mall_restaurant0':'strip mall restaurant0',
        'strip_mall_restaurant10':'strip mall restaurant10',
        'largeoffice_nodatacenter': 'large office no datacenter',
        'strip_mall_restaurant40':'strip mall restaurant40',
        'largeoffice_datacenter': 'large office datacenter',
        'mediumoffice_datacenter': 'medium office datacenter',
        None: 'none'
    }
    datatypes = [
        'string', 'string', 'float', 'float', 'string', 'float', 'float', 'float', 'float', 'float', 'float',
        'float', 'float'
    ]

    savedir = SYSCAPS_PATH / 'captions' / 'comstock'

    idx_files = ['comstock_train_seed=42.idx', 
                 'comstock_hyperparam_train_seed=42.idx', 
                 'comstock_buildings900k_test_seed=42.idx', 
                 'comstock_attribute_combos_seed=42.idx']

    #idx_files = ['comstock_attribute_combos_seed=42.idx']

    for idxf in idx_files:
            
        savedir_ = savedir / 'basic'
        savedir_tokens = savedir / 'basic_tokens' / f'{tok_type}'

        if not savedir_.exists():
            os.makedirs(savedir_)
        if not savedir_tokens.exists():
            os.makedirs(savedir_tokens)

        dataset = EnergyPlusDataset(
                        buildings_bench_path=SYSCAPS_PATH,
                        index_file=idxf,
                        resstock_comstock='comstock',
                        syscaps_split='short', # ignore
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
                    if attributes[i] == 'building type':
                        attr_str += f'{attributes[i]}:{building_types[attrs[j,i]]}|'
                    elif attributes[i] == 'building subtype':
                        attr_str += f'{attributes[i]}:{building_subtypes[attrs[j,i]]}|'
                    else:
                        attr_str += f'{attributes[i]}:{attrs[j,i]}|'
                captions += [attr_str[:-1]]

            encoded_captions = tokenizer(captions, padding=False, truncation=False) 
            #print(encoded_captions['input_ids'][0])
            
            #if idxf == 'comstock_train_seed=42.idx' and cs == 'long':
            #    for ec in encoded_captions['input_ids']:
            #        assert len(ec) > 2, ec

            for bldg_id,c in zip(building_ids, captions):
                with open( savedir_ / f'{bldg_id}_cap.txt', 'w') as f:
                    f.write(c)
            # save tokenized_captions
            for bldg_id,c in zip(building_ids,encoded_captions['input_ids']):
                np.save( savedir_tokens / f'{bldg_id}_cap_ids.npy', c)