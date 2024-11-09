from transformers import DistilBertTokenizer
from syscaps.data.energyplus import EnergyPlusDataset
from syscaps import utils
from pathlib import Path
import os 
import torch
from tqdm import tqdm
import numpy as np

if __name__ == '__main__':

    # llama2-7b-chat , "Return A Similar Building" 5-shot prompt result:
    building_types = {
        'FullServiceRestaurant': 'FineDiningRestaurant',
        'RetailStripmall': 'ShoppingCenter',
        'Warehouse': 'Big Box Store',
        'RetailStandalone': 'ConvenienceStore',
        'SmallOffice': 'Co-WorkingSpace',
        'PrimarySchool': 'ElementarySchool',
        'MediumOffice': 'Workplace',
        'SecondarySchool': 'HighSchool',
        'Outpatient': 'MedicalClinic',
        'QuickServiceRestaurant': 'FastFoodRestaurant',
        'LargeOffice': 'OfficeTower',
        'LargeHotel': 'Five-Star Hotel',
        'SmallHotel': 'Bed and Breakfast',
        'Hospital': 'HealthcareFacility'
    }


    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

    ## Env variables
    SYSCAPS_PATH = os.environ.get('SYSCAPS', '')
    if SYSCAPS_PATH == '':
        raise ValueError('SYSCAPS environment variable not set')

    attributes = open(SYSCAPS_PATH / 'metadata' / 'attributes_comstock.txt', 'r').read().split('\n')
    attributes = [x.strip('"') for x in attributes]

    savedir = SYSCAPS_PATH / 'captions' / 'comstock'

    # test buildings
    idx_files = ['comstock_val_seed=42.idx',
                'comstock_buildings900k_test_seed=42.idx', 
                 'comstock_attribute_combos_seed=42.idx']
    
    #caption_splits = ['long', 'medium', 'short']
    #caption_splits = ['short']
    for idxf in idx_files:
        #for cs in caption_splits:
            
        #print(f'idx file {idxf}, caption split {cs}')
        savedir_ = savedir / f'basic_building_types_cat'
        savedir_tokens = savedir / f'basic_building_types_cat_tokens' / 'distilbert-base-uncased'

        if not savedir_.exists():
            os.makedirs(savedir_)
        if not savedir_tokens.exists():
            os.makedirs(savedir_tokens)

        dataset = EnergyPlusDataset(
                        buildings_bench_path=Path(SYSCAPS_PATH),
                        index_file=idxf,
                        resstock_comstock='comstock',
                        syscaps_split='basic',
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
            onehot = batch['building_attributes_onehot']
            onehot_np = onehot.numpy()
            attrs = dataset.attribute_onehot_encoder.inverse_transform(onehot_np[:,0])

            captions = []
            for j in range(attrs.shape[0]):
                attr_str = ''
                for i in range(len(attributes)):
                    if attributes[i].replace("in.","") == 'building_type':
                        #new_attr = building_types[ attrs[j,i] ]
                        new_attr = 'cat'
                    else:
                        new_attr = attrs[j,i]
                    attr_str += f'{attributes[i].replace("in.","")}:{new_attr}|'
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