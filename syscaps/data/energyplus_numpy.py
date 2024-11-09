import torch
import numpy as np
import datetime
from pathlib import Path
import pyarrow.parquet as pq
import syscaps.transforms as transforms
from syscaps.transforms import BoxCoxTransform, StandardScalerTransform
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
import lightgbm as lgb


class EnergyPlusNumpy(torch.utils.data.Dataset):
    r"""
    Dataset for EnergyPlus data, indexed by hours across all buildings. You can either retrieve all hours (8759 hours) from all buildings,
    or a random subset of hours (e.g., 438 hours) from all buildings.
    """
    def __init__(self, 
                data_path: Path,
                index_file: str,
                resstock_comstock : str = 'comstock',
                return_full_year: bool = True,
                hours_per_building: int = None,
                ordinal_encode: bool = False):
        """
        Args:
            data_path (Path): Path to the pretraining dataset.
            index_file (str): Name of the index file
            return_full_year (bool, optional): Return the full year of hours for each building. Defaults to True.
            hours_per_building (int, optional): Number of hours to return per building. 
                If this is None and return_full_year=False, returns 438 hours per building by default.
            ordinal_encode (bool): If true, output ordinal encoded values, otherwise one-hot encoded values. Default = Fasle.
        """
        BB_split = 'Buildings-900K-test' if 'buildings900k_test' in index_file \
            else 'Buildings-900K/end-use-load-profiles-for-us-building-stock'
        self.buildings_bench_path = data_path / BB_split / '2021'
        self.captions_path = data_path / 'captions'
        self.metadata_path = data_path / 'metadata'

        
        self.building_type_and_year = ['comstock_tmy3_release_1',
                                       'resstock_tmy3_release_1',
                                       'comstock_amy2018_release_1',
                                       'resstock_amy2018_release_1']
        self.census_regions = ['by_puma_midwest', 'by_puma_south', 'by_puma_northeast', 'by_puma_west']

        self.index_file = self.metadata_path / 'syscaps' / 'splits' / index_file
        self.index_fp = None
        self.__read_index_file(self.index_file)

        self.time_transform = transforms.TimestampTransform()
        self.resstock_comstock = resstock_comstock
        self.return_full_year = return_full_year
        
        if self.resstock_comstock == 'comstock':
            self.attributes = open(self.metadata_path / 'syscaps' / 'energyplus' / 'attributes_comstock.txt', 'r').read().strip().split('\n')
            df1 = pd.read_parquet(self.buildings_bench_path / 'comstock_amy2018_release_1' / 'metadata' / 'metadata.parquet', engine="pyarrow")
            df2 = pd.read_parquet(self.buildings_bench_path / 'comstock_tmy3_release_1' / 'metadata' / 'metadata.parquet', engine="pyarrow")
            self.attribute_dfs = {
                'comstock_amy2018_release_1': df1,
                'comstock_tmy3_release_1': df2
            }
        else:
            self.attributes = open(self.metadata_path / 'syscaps' / 'energyplus' / 'attributes_resstock.txt', 'r').read().strip().split('\n')
            df1 = pd.read_parquet(self.buildings_bench_path / 'resstock_amy2018_release_1' / 'metadata' / 'metadata.parquet', engine="pyarrow")
            df2 = pd.read_parquet(self.buildings_bench_path / 'resstock_tmy3_release_1' / 'metadata' / 'metadata.parquet', engine="pyarrow")
            self.attribute_dfs = {
                'resstock_amy2018_release_1': df1,
                'resstock_tmy3_release_1': df2
            }
        self.attributes = [x.strip('"') for x in self.attributes]
        self.attributes = [x for x in self.attributes if x != ""] # remove empty string

        #df = pd.concat([df1, df2])
        df = df1.loc[ df1.index.intersection(df2.index).values ]

        self.num_attributes = len(self.attributes)

        if ordinal_encode:
            # use ordinal encoder
            self.attribute_ordinal_encoder = OrdinalEncoder()
            self.attribute_ordinal_encoder.fit(df[self.attributes].values)
            print('Used ordinal encoder')
        else:
            self.attribute_onehot_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
            self.attribute_onehot_encoder.fit(df[self.attributes].values)
            # Example: the sum of possible values for comstock attrs is 336. so the encoding vector
            # is of dim 336 (although it is really 13 onehot vectors concated together.)
            self.onehot_shape = self.attribute_onehot_encoder.transform([[None] * self.num_attributes]).shape
            print(f'OneHotEncoder: # attributes {self.num_attributes}, attributes onehot shape = {self.onehot_shape}')

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
        self.load_transform.load(self.metadata_path / 'syscaps' / 'transforms' / resstock_comstock / 'load')

        self.weather_transforms = []           
        for col in self.weather_feature_names[1:]:
            self.weather_transforms += [ StandardScalerTransform() ]
            self.weather_transforms[-1].load(self.metadata_path / 'transforms' / 'weather' / col)
        if not self.return_full_year:
            print('[WARNING] One epoch of the EnergyPlusDataset in non-sequential mode is'
              ' only 1 hour/building')

        if self.return_full_year:
            self.hours_per_building = 8759
        elif hours_per_building is None:
            self.hours_per_building = 438 ## ~5% of the year
        else:
            self.hours_per_building = hours_per_building

        self.ordinal_encode = ordinal_encode

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
        return self.num_buildings * self.hours_per_building

    
    def get_sample(self, 
                   bldg_df: pd.DataFrame,
                   bldg_id: str,
                   attribute_df: pd.DataFrame,
                   weather_df: pd.DataFrame,
                   hour: int):
        
        # Slice each column from seq_ptr-context_len : seq_ptr + pred_len
        time_features = self.time_transform.transform(bldg_df['timestamp'].iloc[hour : hour + 1])
        load_features = bldg_df[bldg_id].iloc[hour : hour + 1].values.astype(np.float32)

        # For BoxCox transform
        load_features = self.load_transform.transform(load_features)

        # one-hot encode attributes
        atts = attribute_df.loc[int(bldg_id)][self.attributes].values
        if self.ordinal_encode:
            atts_values = self.attribute_ordinal_encoder.transform(np.reshape(atts, (1,self.num_attributes)))
            atts_values = atts_values[0]
        else:
            # one-hot encoded attributes
            # shape is [1, sum(num_possible_values)], Ex: 336 for comstock
            atts_values = self.attribute_onehot_encoder.transform(np.reshape(atts, (1,self.num_attributes)))
            atts_values = atts_values[0]

        ## weather                    
        weather_df = weather_df.iloc[hour : hour + 1] 
        
        # convert temperature to fahrenheit (note: keep celsius for now)
        # weather_df['temperature'] = weather_df['temperature'].apply(lambda x: x * 1.8 + 32) 

        # weather transform
        weather_arr = []
        for idx,col in enumerate(weather_df.columns[1:]):
            ## WARNING: CONVERTS TO TORCH FROM NUMPY AUTOMATICALLY
            weather_arr.append(self.weather_transforms[idx].transform(weather_df[col].to_numpy())[0][0].item())

        sample = np.hstack((time_features[0], atts_values, weather_arr, load_features[0]))
        return sample


    def __getitem__(self, idx):
        # Open file pointer if not already open
        if not self.index_fp:
           self.index_fp = open(self.index_file, 'rb', buffering=0)
           self.index_fp.seek(0)

        # Get the index of the time series
        self.index_fp.seek(int(idx / self.hours_per_building) * self.chunk_size, 0)
        ts_idx = self.index_fp.read(self.chunk_size).decode('utf-8')

        # Parse the index
        ts_idx = ts_idx.strip('\n').split('\t')
        bldg_id = ts_idx[3].lstrip('0')
        dataset_id = int(ts_idx[0])
        puma       = ts_idx[2]

        # Select timestamp and building column
        df = pq.read_table(str(self.buildings_bench_path / self.building_type_and_year[dataset_id]
                        / 'timeseries_individual_buildings' / self.census_regions[int(ts_idx[1])]
                        / 'upgrade=0' / f'puma={puma}'), columns=['timestamp', bldg_id])

        # Order by timestamp
        df = df.to_pandas().sort_values(by='timestamp')
        
        # get county ID
        county = self.weather_lookup_df.loc[puma]['nhgis_2010_county_gisjoin']
        # load corresponding weather files
        weather_df = pd.read_csv(str(self.buildings_bench_path / self.building_type_and_year[dataset_id] / 'weather' / f'{county}.csv'))
        assert datetime.datetime.strptime(weather_df['date_time'].iloc[0], '%Y-%m-%d %H:%M:%S').strftime('%m-%d') == '01-01',\
            "The weather file does not start from Jan 1st"      
        weather_df.columns = self.weather_feature_names   
        weather_df = weather_df[self.weather_feature_names]
        weather_df = weather_df.iloc[:-1] # remove last hour to align with load data

        if self.return_full_year or self.hours_per_building == 8759:
            hour = idx % self.hours_per_building
        else:
            hour = np.random.randint(0, 8759)
            
        sample = self.get_sample(df, 
                                 bldg_id,
                                 self.attribute_dfs[self.building_type_and_year[dataset_id]],
                                 weather_df,
                                 hour)

        return sample

