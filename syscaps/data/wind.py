import torch 
from pathlib import Path 
import h5py
import random 
import numpy as np
from transformers import  DistilBertTokenizer, BertTokenizer
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

class WindDataset(torch.utils.data.Dataset):
    r"""This is a dataset for loading wind data for surrogate modeling.

    the index file is a tab separated file with the following columns:

    layout_id scenario_id 

    and specifies the scenarios for train/val/test.
    """
    def __init__(self, 
                data_path: Path,
                index_file: str,
                syscaps_split : str = 'basic',
                use_random_caption_augmentation: bool = True,
                caption_augmentation_style: int = 1,
                tokenizer = 'distilbert-base-uncased',
                include_text: bool = False):
        self.captions_path = data_path / 'captions'
        self.metadata_path = data_path / 'metadata'
        self.include_text = include_text
        self.tokenizer_name = tokenizer
        if self.tokenizer_name == "distilbert-base-uncased":
            self.tokenizer = DistilBertTokenizer.from_pretrained(self.tokenizer_name)
        elif self.tokenizer_name == "bert-base-uncased":
            self.tokenizer = BertTokenizer.from_pretrained(self.tokenizer_name)
        
        self.syscaps_split = syscaps_split
        self.use_random_caption_augmentation = use_random_caption_augmentation
        self.caption_augmentation_style = caption_augmentation_style
        
        if self.syscaps_split == 'medium' and self.include_text and self.use_random_caption_augmentation:
            self.captions_data = {}
            for i in range(4):
                self.captions_data[i] = pd.read_csv(self.captions_path / 'wind' / \
                    self.syscaps_split / f'aug_{i}' / 'captions.csv',
                    header=0, index_col=0)
        elif self.syscaps_split == 'medium' and self.include_text:
            self.captions_data = pd.read_csv(self.captions_path / 'wind' / \
                self.syscaps_split / f'aug_{self.caption_augmentation_style}' / 'captions.csv', header=0, index_col=0)
        elif self.include_text:
            self.captions_data = pd.read_csv(self.captions_path / 'wind' / \
                self.syscaps_split / 'captions.csv', header=0, index_col=0)
            
        self.index_file = self.metadata_path / 'syscaps' / 'splits' / index_file
        self.index_fp = None
        self.__read_index_file(self.index_file)

        self.h5_data =  h5py.File(self.metadata_path / 'syscaps' / 'wind' / 'wind_plant_data.h5', 'r')
        self.layout_types = pd.read_csv(self.metadata_path / 'syscaps' / 'wind' / 'layout_type.csv')
        self.mean_rotor_diameters = pd.read_csv(self.metadata_path / 'syscaps' / 'wind' /'results_mean_turbine_spacing.txt', header=None)

        self.target_transform = lambda x: x / (675.195904 - 0) # max-min 
        self.undo_transform = lambda x: x * (675.195904 - 0)
        
        self.attribute_onehot_encoder = OneHotEncoder(
            handle_unknown='ignore', sparse=False,
            categories=[['cluster', 'single string', 'parallel strings', 'multiple strings'],
                        list(range(25,201)),
                        list(range(3,10)),
                        ])
        self.attribute_onehot_encoder.fit(np.array([None] * 3).reshape(-1,3))
        self.onehot_shape = self.attribute_onehot_encoder.transform([[None] * 3]).shape
        print(f'OneHotEncoder: # attributes 3, attributes onehot shape = {self.onehot_shape}')
  
    def init_fp(self):
        """Each worker needs to open its own file pointer to avoid 
        weird multiprocessing errors from sharing a file pointer.

        This is not called in the main process.
        This is called in the DataLoader worker_init_fn.
        The file is opened in binary mode which lets us disable buffering.
        """
        self.index_fp = open(self.index_file, 'rb', buffering=0)
        self.index_fp.seek(0)

    def __read_index_file(self, index_file: Path) -> None:
        """Extract metadata from index file.
        """        
        def _count_generator(reader):
            b = reader(1024 * 1024)
            while b:
                yield b
                b = reader(1024 * 1024)

        with open(index_file, 'rb') as fp:
            c_generator = _count_generator(fp.raw.read)
            # count each \n
            self.num_datapoints = sum(buffer.count(b'\n') for buffer in c_generator)
        
        # Count the number of chars per line
        with open(index_file, 'rb', buffering=0) as fp:
            first_line = fp.readline()
            self.chunk_size = len(first_line)
        

    def __del__(self):
        if self.index_fp:
            self.index_fp.close()   
        self.h5_data.close()

    def __len__(self):
        return self.num_datapoints
    
    def __getitem__(self, idx):

        # wind speed (0-25 m/s)
        # wind direction (0-360 degrees)
        # turbulence intensity 6, 8, 10

        # Open file pointer if not already open
        if not self.index_fp:
           self.index_fp = open(self.index_file, 'rb', buffering=0)
           self.index_fp.seek(0)
        # Get the index of the time series
        self.index_fp.seek(idx * self.chunk_size, 0)
        row = self.index_fp.read(self.chunk_size).decode('utf-8')
        layout_id, scenario_id = row.strip('\n').split('\t')

        # Load the data
        y = np.array([np.sum(self.h5_data[layout_id]['Scenarios'][scenario_id]['Turbine Power'][:]) / 1e6])
        y = self.target_transform(y)

        # Load the atmospheric variables
        wind_speed = self.h5_data[layout_id]['Scenarios'][scenario_id]['Wind Speed'][()]
        wind_speed = np.array([wind_speed]) / 25.0 # normalize
        wind_direction = self.h5_data[layout_id]['Scenarios'][scenario_id]['Wind Direction'][()]
        wind_direction = np.array([wind_direction]) / 360.0
        turbulence_intensity = self.h5_data[layout_id]['Scenarios'][scenario_id]['Turbulence Intensity'][()]
        turbulence_intensity = (100*np.array([turbulence_intensity])) / 10.0

        sample = {
            'layout_id': layout_id,
            'scenario_id': scenario_id,
            'wind_speed': wind_speed,
            'wind_direction': wind_direction,
            'turbulence_intensity': turbulence_intensity,
            'y': y
        }
        # load one-hot attributes
        layout_num = int(layout_id.split('Layout')[1])
        atts = [
            self.layout_types.iloc[layout_num]['Layout Type'],
            self.h5_data[layout_id]['Number of Turbines'][()],
            self.mean_rotor_diameters.iloc[layout_num].values[0]
        ]
        onehots = self.attribute_onehot_encoder.transform([atts])
        sample['attributes_onehot']    = onehots.astype(np.float32)

        if self.syscaps_split == 'medium' and \
            self.use_random_caption_augmentation:
            # randomly sample an augmentation style 
            style = random.choice([0,1,2,3])
            #style = 1
            caption_data = self.captions_data[style]
            caption_tokens_dir = self.captions_path / 'wind' / \
                self.syscaps_split / \
                    f'aug_{style}_tokens' / self.tokenizer_name
        elif self.syscaps_split == 'medium' and \
            not self.use_random_caption_augmentation:
            style = self.caption_augmentation_style
            caption_data = self.captions_data
            caption_tokens_dir = self.captions_path / 'wind' / \
                self.syscaps_split / \
                    f'aug_{style}_tokens' / self.tokenizer_name
        else: # basic
            caption_data = self.captions_data
            caption_tokens_dir = self.captions_path / 'wind' / \
                f'{self.syscaps_split}_tokens' / \
                self.tokenizer_name
            
        if self.include_text:
            sample['syscaps'] = caption_data.loc[layout_id, 'caption']
                
        tokenized_caption_path = caption_tokens_dir / f'{layout_id}_cap_ids.npy'

        if (tokenized_caption_path).exists():
            # numpy int32 array 
            sample['attributes_input_ids'] = \
                np.load(tokenized_caption_path, allow_pickle=True).astype(np.int32)
        return sample 
    
    def collate_fn(self):
        """
        Returns a function taking only one argument
        (the list of items to be batched) with the proper
        pad_token_id
        """
        def _collate(samples):

            # day of year, day of week, hour of day
            # load
            batch = {
                'wind_speed': torch.stack([torch.from_numpy(s['wind_speed']) for s in samples]).float(),
                'wind_direction': torch.stack([torch.from_numpy(s['wind_direction']) for s in samples]).float(),
                'turbulence_intensity': torch.stack([torch.from_numpy(s['turbulence_intensity']) for s in samples]).float(),
                'y': torch.stack([torch.from_numpy(s['y']) for s in samples]).float().reshape(-1,1),
                'attributes_onehot': torch.stack([torch.from_numpy(s['attributes_onehot']) for s in samples]).float(),
            }
            
            if 'attributes_input_ids' in samples[0]:
                pad_seq = torch.nn.utils.rnn.pad_sequence(
                        [torch.from_numpy(s['attributes_input_ids']).long() for s in samples],
                        batch_first=True,
                        padding_value=self.tokenizer.pad_token_id)

                att_mask = (pad_seq != self.tokenizer.pad_token_id).long()

                batch['attributes_input_ids'] =  pad_seq
                batch['attributes_attention_mask'] = att_mask
                
            if self.include_text:
                batch['syscaps'] = [s['syscaps'] for s in samples]
                batch['layout_id'] = [s['layout_id'] for s in samples]
                batch['scenario_id'] = [s['scenario_id'] for s in samples]
            
            return batch

        return _collate
    

if __name__ == '__main__':

    import os
    from pathlib import Path
    import torch
    from syscaps import utils
    from tqdm import tqdm

    test_data = WindDataset(
        data_path = Path(os.environ.get('SYSCAPS', '')),
        index_file = 'floris_train_seed=42.idx',
        syscaps_split = 'medium',
        caption_augmentation_style=1,
        include_text = True       
    )

    for i in range(10):
        dat = test_data[i]
        print(dat)

    dataloader = torch.utils.data.DataLoader(
        test_data, batch_size=8, 
        worker_init_fn=utils.worker_init_fn,
        collate_fn=test_data.collate_fn(), num_workers=8)

    for batch in tqdm(dataloader):
        print(batch['turbulence_intensity'])
        for i in range(len(batch['syscaps'])):
            print(batch['syscaps'][i])
            
            print('==========================================')
        break 
