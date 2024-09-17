import torch
import numpy as np
#import datetime
from pathlib import Path
import pyarrow.parquet as pq
import syscaps.transforms as transforms
from syscaps.transforms import BoxCoxTransform, StandardScalerTransform
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from transformers import  DistilBertTokenizer, BertTokenizer, LongformerTokenizer
from typing import List
from datetime import date, datetime, timedelta, time as t


class EnergyPlusDataset(torch.utils.data.Dataset):
    r"""This is a dataset for loading EnergyPlus time series for surrogate modeling.
    Expects buildings from the Buildings-900K dataset.

    the index file is a tab separated file with the following columns:

    0. Building-type-and-year (e.g., comstock_tmy3_release_1)
    1. Census region (e.g., by_puma_midwest)
    2. PUMA ID
    3. Building ID

    Two main modes: non-autoregressive and autoregressive. 

    Non-autoregressive mode (arg: return_full_year = False)
        Randomly sample a single hour from a single random building, at a time. Batches are
        constructed by stacking multiple samples from different buildings.
        The time series are not stored chronologically and must be sorted by timestamp after loading.
        Each dataloader worker has its own file pointer to the index file. This is to avoid
        weird multiprocessing errors from sharing a file pointer. We 'seek' to the correct
        line in the index file for random access.

    Autoregressive mode (arg: return_full_year = True)
        Load a single year of data from a single random building, at a time. Batches are
        constructed by stacking years from different buildings.
        The time series are not stored chronologically and must be sorted by timestamp after loading.
        The entire year is loaded into memory and returned as a sequence of length 8760.    
    """
    def __init__(self, 
                buildings_bench_path: Path,
                index_file: str,
                resstock_comstock : str = 'comstock',
                syscaps_split : str = 'short',
                tokenizer = 'distilbert-base-uncased',
                return_full_year: bool = False,
                include_text: bool = False,
                caption_template = ''):
        """
        Args:
            buildings_bench_path (Path): Path to the pretraining dataset.
            index_file (str): Name of the index file
            # TODO:
        """
        BB_split = 'Buildings-900K-test' if 'buildings900k_test' in index_file \
            else 'Buildings-900K/end-use-load-profiles-for-us-building-stock'
        self.buildings_bench_path = buildings_bench_path / BB_split / '2021'
        self.captions_path = buildings_bench_path / 'captions'
        self.metadata_path = buildings_bench_path / 'metadata'
        self.caption_template = caption_template
        
        self.building_type_and_year = ['comstock_tmy3_release_1',
                                       'resstock_tmy3_release_1',
                                       'comstock_amy2018_release_1',
                                       'resstock_amy2018_release_1']
        self.census_regions = ['by_puma_midwest', 'by_puma_south', 'by_puma_northeast', 'by_puma_west']

        self.index_file = self.metadata_path / 'splits' / index_file
        self.index_fp = None
        self.__read_index_file(self.index_file)

        self.time_transform = transforms.TimestampTransform()
        self.resstock_comstock = resstock_comstock
        self.syscaps_split = syscaps_split
        self.return_full_year = return_full_year
        self.include_text = include_text

        self.tokenizer_name = tokenizer
        if self.tokenizer_name == "distilbert-base-uncased":
            self.tokenizer = DistilBertTokenizer.from_pretrained(self.tokenizer_name)
            print(f'[EnergyPlusDataset] Captions with more than 512 tokens will be truncated!')
        elif self.tokenizer_name == "bert-base-uncased":
            self.tokenizer = BertTokenizer.from_pretrained(self.tokenizer_name)
            print(f'[EnergyPlusDataset] Captions with more than 512 tokens will be truncated!')
        elif self.tokenizer_name == "longformer-base-4096":
            self.tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
            print(f'[EnergyPlusDataset] Using longformer tokenizer, which supports up to 4096 tokens')

        if self.resstock_comstock == 'comstock':
            self.attributes = open(self.metadata_path / 'attributes_comstock.txt', 'r').read().strip().split('\n')
            df1 = pd.read_parquet(self.metadata_path / "comstock_amy2018.parquet", engine="pyarrow")
            df2 = pd.read_parquet(self.metadata_path / "comstock_tmy3.parquet", engine="pyarrow")
            self.attribute_dfs = {
                'comstock_amy2018_release_1': df1,
                'comstock_tmy3_release_1': df2
            }
        else:
            self.attributes = open(self.metadata_path / 'attributes_resstock.txt', 'r').read().strip().split('\n')
            df1 = pd.read_parquet(self.metadata_path / "resstock_amy2018.parquet", engine="pyarrow")
            df2 = pd.read_parquet(self.metadata_path / "resstock_tmy3.parquet", engine="pyarrow")
            self.attribute_dfs = {
                'resstock_amy2018_release_1': df1,
                'resstock_tmy3_release_1': df2
            }
        self.attributes = [x.strip('"') for x in self.attributes]
        # remove empty strings
        self.attributes = list(filter(None, self.attributes))
        df = df1.loc[ df1.index.intersection(df2.index).values ]
        
        self.attribute_onehot_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
        self.attribute_onehot_encoder.fit(df[self.attributes].values)
        
        self.num_attributes = len(self.attributes)
        # Example: the sum of possible values for comstock attrs is 336. so the encoding vector
        # is of dim 336 (although it is really 13 onehot vectors concated together.)
        self.onehot_shape = self.attribute_onehot_encoder.transform([[None] * self.num_attributes]).shape

        print(f'[EnergyPlusDataset] # attributes {self.num_attributes}, attributes onehot shape = {self.onehot_shape}')

        ## Weather lookup
        lookup_df = pd.read_csv(self.metadata_path / 'spatial_tract_lookup_table.csv')
        # select rows that have weather
        df_has_weather = lookup_df[(lookup_df.weather_file_2012 != 'No weather file') 
                                    & (lookup_df.weather_file_2015 != 'No weather file') 
                                    & (lookup_df.weather_file_2016 != 'No weather file') 
                                    & (lookup_df.weather_file_2017 != 'No weather file') 
                                    & (lookup_df.weather_file_2018 != 'No weather file') 
                                    & (lookup_df.weather_file_2019 != 'No weather file')]

        df_has_weather = df_has_weather[['nhgis_2010_county_gisjoin', 'nhgis_2010_puma_gisjoin']]
        df_has_weather = df_has_weather.set_index('nhgis_2010_puma_gisjoin')
        self.weather_lookup_df = df_has_weather[~df_has_weather.index.duplicated()] # remove duplicated indices
        self.weather_feature_names = ['timestamp', 'temperature', 'humidity', 'wind_speed', 'wind_direction', 'global_horizontal_radiation', 
                        'direct_normal_radiation', 'diffuse_horizontal_radiation']
                                    
        self.load_transform = BoxCoxTransform()
        self.load_transform.load(self.metadata_path / 'transforms' / resstock_comstock / 'load')

        self.weather_transforms = []           
        for col in self.weather_feature_names[1:]:
            self.weather_transforms += [ StandardScalerTransform() ]
            self.weather_transforms[-1].load(self.metadata_path / 'transforms' / 'weather' / col)
        if not self.return_full_year:
            print('[EnergyPlusDataset] Each epoch of the EnergyPlusDataset in non-sequential mode means'
              ' your model sees 1 hour per building')
            
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
        # Fast solution to get the number of time series in index file
        # https://pynative.com/python-count-number-of-lines-in-file/
        
        def _count_generator(reader):
            b = reader(1024 * 1024)
            while b:
                yield b
                b = reader(1024 * 1024)

        with open(index_file, 'rb') as fp:
            c_generator = _count_generator(fp.raw.read)
            # count each \n
            self.num_buildings = sum(buffer.count(b'\n') for buffer in c_generator)
        
        # Count the number of chars per line
        with open(index_file, 'rb', buffering=0) as fp:
            first_line = fp.readline()
            self.chunk_size = len(first_line)
        
        #print(f'Counted {self.num_buildings} indices in index file.')

    def __del__(self):
        if self.index_fp:
            self.index_fp.close()

    def __len__(self):
        return self.num_buildings

    
    def get_sample(self, 
                   bldg_df: pd.DataFrame,
                   bldg_id: str,
                   attribute_df: pd.DataFrame,
                   weather_df: pd.DataFrame,
                   hour_start: int,
                   hour_end: int):
        
        # Slice each column from seq_ptr-context_len : seq_ptr + pred_len
        time_features = self.time_transform.transform(bldg_df['timestamp'].iloc[hour_start : hour_end ])
        load_features = bldg_df[bldg_id].iloc[ hour_start : hour_end ].values.astype(np.float32)

        # For BoxCox transform
        load_features = self.load_transform.transform(load_features)

        sample = {
            'day_of_year': time_features[:, 0][...,None],
            'day_of_week': time_features[:, 1][...,None],
            'hour_of_day': time_features[:, 2][...,None],
            'y': load_features[...,None]
        }

        # one-hot encode attributes
        atts = attribute_df.loc[int(bldg_id)][self.attributes].values
        # shape is [1, sum(num_possible_values)], Ex: 336 for comstock
        onehots = self.attribute_onehot_encoder.transform(np.reshape(atts, (1,self.num_attributes)))

        sample['attributes_onehot']    = onehots.astype(np.float32)
        sample['building_id']                   = bldg_id
        #sample['dataset_id']                    = int(dataset_id)
       
        syscaps_split = self.syscaps_split

        if self.include_text:
            with open(self.captions_path /  self.resstock_comstock / \
                syscaps_split / f'{bldg_id}_cap.txt') as f:
                sample['syscaps'] = f.read() # string
                # Just for evaluation purposes
                if (self.captions_path /  self.resstock_comstock / \
                syscaps_split / f'{bldg_id}_cap_attribute.txt').exists():
                    with open(self.captions_path /  self.resstock_comstock / \
                        syscaps_split / f'{bldg_id}_cap_attribute.txt') as f:
                        sample['syscaps_missing'] = f.read().strip()
                elif (self.captions_path /  self.resstock_comstock / \
                (syscaps_split + '_missing') / f'{bldg_id}_cap_attribute.txt').exists():
                    with open(self.captions_path /  self.resstock_comstock / \
                        (syscaps_split + '_missing') / f'{bldg_id}_cap_attribute.txt') as f:
                        sample['syscaps_missing'] = f.read().strip()

        tokenized_caption_path = self.captions_path / self.resstock_comstock / \
                    f'{syscaps_split}_tokens' / \
                        self.tokenizer_name / f"{bldg_id}_cap_ids.npy"
        if (tokenized_caption_path).exists():
            # numpy int32 array 
            sample['attributes_input_ids'] = \
                np.load(tokenized_caption_path, allow_pickle=True).astype(np.int32)
            # truncate to 512 tokens for Bert-style models
            if "bert" in self.tokenizer_name:
                sample['attributes_input_ids'] = sample['attributes_input_ids'][:512]

        ## weather                    
        weather_df = weather_df.iloc[hour_start : hour_end] 
        
        # convert temperature to fahrenheit (note: keep celsius for now)
        # weather_df['temperature'] = weather_df['temperature'].apply(lambda x: x * 1.8 + 32) 

        # weather transform
        for idx,col in enumerate(weather_df.columns[1:]):
            ## WARNING: CONVERTS TO TORCH FROM NUMPY AUTOMATICALLY
            sample.update({col : self.weather_transforms[idx].transform(
                weather_df[col].to_numpy())[0][...,None]})

        return sample


    def __getitem__(self, idx):
        # Open file pointer if not already open
        if not self.index_fp:
           self.index_fp = open(self.index_file, 'rb', buffering=0)
           self.index_fp.seek(0)

        # Get the index of the time series
        self.index_fp.seek(idx * self.chunk_size, 0)
        ts_idx = self.index_fp.read(self.chunk_size).decode('utf-8')

        # Parse the index
        ts_idx     = ts_idx.strip('\n').split('\t')
        bldg_id    = ts_idx[3].lstrip('0')
        dataset_id = int(ts_idx[0])
        puma       = ts_idx[2]

        # Select timestamp and building column
        df = pq.read_table(str(self.buildings_bench_path / self.building_type_and_year[dataset_id]
                        / 'timeseries_individual_buildings' / self.census_regions[int(ts_idx[1])]
                        / 'upgrade=0' / f'puma={puma}'), columns=['timestamp', bldg_id])

        # Order by timestamp
        df = df.to_pandas().sort_values(by='timestamp')
        
        ### load single hourly data
        if not self.return_full_year:    
            # strip leading zeros
            #hour = ts_idx[-1]
            #hour = int(hour.lstrip('0')) if hour != '0000' else 0

            # sample a random number between 0 and 8759
            hour_start = np.random.randint(0, 8759)
            hour_end = hour_start + 1
        else: # load a full year of data...
            hour_start = 0
            hour_end = 8759 
            assert hour_end <= len(df), "The time series is not long enough for autoregressive mode."
        
        # get county ID
        county = self.weather_lookup_df.loc[puma]['nhgis_2010_county_gisjoin']
        # load corresponding weather files
        weather_df = pd.read_csv(str(self.buildings_bench_path / self.building_type_and_year[dataset_id] / 'weather' / f'{county}.csv'))
        assert datetime.strptime(weather_df['date_time'].iloc[0], '%Y-%m-%d %H:%M:%S').strftime('%m-%d') == '01-01',\
            "The weather file does not start from Jan 1st"      
        weather_df.columns = self.weather_feature_names   
        weather_df = weather_df[self.weather_feature_names]
        weather_df = weather_df.iloc[:-1] # remove last hour to align with load data

        sample = self.get_sample(df, 
                                 bldg_id,
                                 self.attribute_dfs[self.building_type_and_year[dataset_id]],
                                 weather_df,
                                 hour_start, hour_end)

        return sample

    def process_attributes_to_caption(self, attribute_name, attribute_value, opening_time=None):
        
        building_types = {
            'FullServiceRestaurant': 'full service restaurant',
            'RetailStripmall': 'strip mall restaurant',
            'Warehouse': 'warehouse',
            'RetailStandalone': 'retail standalone',
            'SmallOffice': 'small-sized office',
            'PrimarySchool': 'primary school',
            'MediumOffice': 'medium-sized office',
            'SecondarySchool': 'secondary school',
            'Outpatient': 'outpatient',
            'QuickServiceRestaurant': 'quick-service restaurant',
            'LargeOffice': 'large-sized office',
            'LargeHotel': 'large-sized hotel',
            'SmallHotel': 'small-sized hotel',
            'Hospital': 'hospital'
        }
        
        if attribute_name == 'in.building_type':
            return building_types[attribute_value]
        
        elif 'opening_time' in attribute_name:
            m = attribute_value * 60
            h = int(m // 60)
            m = int(m % 60)
            attribute_value = t(hour=h, minute=m)
            v = attribute_value
            attribute_value = "%d:%02d" % (v.hour, v.minute)
            attribute_value += " AM" if v.hour < 12 else " PM"  
            return attribute_value
        elif 'operating_hours' in attribute_name:
            m = attribute_value * 60
            h = int(m // 60)
            m = int(m % 60)
            v = timedelta(hours=h, minutes=m)
            m = opening_time * 60
            h = int(m // 60)
            m = int(m % 60)
            opening_time_v = t(hour=h, minute=m)
            attribute_value = (datetime.combine(date.today(), opening_time_v) + v).time()
            v=attribute_value
            attribute_value = "%d:%02d" % (v.hour, v.minute)
            attribute_value += " AM" if v.hour < 12 else " PM"  
            return attribute_value
        elif 'sqft' in attribute_name:
            return f'{attribute_value:,}'
        elif 'stories' in attribute_name:
            if int(attribute_value) == 1:
                return 'single-story'
            elif attribute_value < 11:
                # convert to words
                return ['', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten'][int(attribute_value)-1] + '-story'
            else:
                return str(int(attribute_value)) + '-story'
        else:
            return attribute_value 
           

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
                'day_of_year': torch.stack([torch.from_numpy(s['day_of_year']) for s in samples]).float(),
                'day_of_week': torch.stack([torch.from_numpy(s['day_of_week']) for s in samples]).float(),
                'hour_of_day': torch.stack([torch.from_numpy(s['hour_of_day']) for s in samples]).float(),
                'y': torch.stack([torch.from_numpy(s['y']) for s in samples]).float(),
                'attributes_onehot': torch.stack([torch.from_numpy(s['attributes_onehot']) for s in samples]).float(),
            }
            # weather features
            for w in self.weather_feature_names[1:]:
                batch[w] = torch.stack([s[w] for s in samples]).float()

            if self.caption_template != '':
                batch['syscaps'] = []

                # on the fly, generate caption based on template, tokenize it.
                #new_captions = []
                words = self.caption_template.lower().split()
                template_lookup = {}
                for idx,w in enumerate(words):
                    if w in self.attributes:
                        template_lookup[w] = (idx,self.attributes.index(w))

                # replace any words that match attribute names with attribue values
                for s in samples:
                    words_ = self.caption_template.lower().split()
                    attrs = self.attribute_onehot_encoder.inverse_transform(s['attributes_onehot'])[0]
                    for k,v in template_lookup.items():
                        word_idx, attr_idx = v[0], v[1]
                        if self.attributes[attr_idx] == 'in.weekday_operating_hours':
                            words_[word_idx] = str( self.process_attributes_to_caption(
                                self.attributes[attr_idx], 
                                attrs[attr_idx],
                                attrs[self.attributes.index('in.weekday_opening_time')]) )
                        elif self.attributes[attr_idx] == 'in.weekend_operating_hours':
                            words_[word_idx] = str( self.process_attributes_to_caption(
                                self.attributes[attr_idx], attrs[attr_idx],
                                attrs[self.attributes.index('in.weekend_opening_time')]) )
                        else:
                            words_[word_idx] = str( self.process_attributes_to_caption(self.attributes[attr_idx], attrs[attr_idx]) )
                    batch['syscaps'].append(' '.join(words_))

                batch['attributes_input_ids'], batch['attributes_attention_mask'] = self.tokenizer.batch_encode_plus(
                    batch['syscaps'], 
                    padding=True, 
                    return_attention_mask=True, 
                    return_tensors='pt').values()

                batch['building_id'] = [s['building_id'] for s in samples]
            else:
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
                    batch['building_id'] = [s['building_id'] for s in samples]
                    if 'syscaps_missing' in samples[0]:
                        batch['syscaps_missing'] = [s['syscaps_missing'] for s in samples]
            return batch

        return _collate



if __name__ == '__main__':
    import os
    from pathlib import Path
    import torch
    from syscaps import utils
    from tqdm import tqdm
    from transformers import  DistilBertTokenizer

    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    test_data = EnergyPlusDataset(
        buildings_bench_path = Path(os.environ.get('BUILDINGS_BENCH', '')),
        index_file = 'resstock_train_seed=42.idx',
        resstock_comstock = 'resstock',
        syscaps_split = 'medium',
        return_full_year = False,
        include_text = True       
    )

    for i in range(4):
        dat = test_data[i]
        cap = tokenizer.decode(dat['attributes_input_ids'])
        print(cap)

    # dataloader = torch.utils.data.DataLoader(
    #     test_data, batch_size=8, 
    #     worker_init_fn=utils.worker_init_fn,
    #     collate_fn=test_data.collate_fn(), num_workers=8)

    # for batch in tqdm(dataloader):

    #     for i in range(len(batch['syscaps'])):
    #         print(batch['syscaps'][i])
    #         print('==========================================')
    #     break 
        
    