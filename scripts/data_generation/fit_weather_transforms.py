import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import random
import os
from tqdm import trange, tqdm 
from syscaps.transforms import StandardScalerTransform


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)

    project_dir = Path(os.environ.get('SYSCAPS', ''))
    output_dir = project_dir / 'metadata' / 'transforms' / 'weather'
    buildings_for_transform = project_dir / 'metadata' / 'splits' / args.energyplus_index_file    

    print(f'Fitting weather transform for {str(buildings_for_transform)}')

    # training set dir
    time_series_dir = Path(os.environ.get('SYSCAPS', ''), 'Buildings-900K', 'end-use-load-profiles-for-us-building-stock', '2021')    
    building_years = ['comstock_tmy3_release_1', 'resstock_tmy3_release_1', 'comstock_amy2018_release_1', 'resstock_amy2018_release_1'] 

    bldgs_df = pd.read_csv(buildings_for_transform, sep='\t', names=['building_years', 'census', 'puma', 'bldg_id'])

    weather_columns = ['timestamp', 'temperature', 'humidity', 'wind_speed', 'wind_direction', 'global_horizontal_radiation', 
                              'direct_normal_radiation', 'diffuse_horizontal_radiation']
    #weather_df = pd.DataFrame(columns=weather_columns)
    weather_data = {}
    for wc in weather_columns[1:]:
        weather_data[wc] = []

    ## Weather lookup
    lookup_df = pd.read_csv(project_dir / 'metadata' / 'spatial_tract_lookup_table.csv')
    # select rows that have weather
    df_has_weather = lookup_df[(lookup_df.weather_file_2012 != 'No weather file') 
                                & (lookup_df.weather_file_2015 != 'No weather file') 
                                & (lookup_df.weather_file_2016 != 'No weather file') 
                                & (lookup_df.weather_file_2017 != 'No weather file') 
                                & (lookup_df.weather_file_2018 != 'No weather file') 
                                & (lookup_df.weather_file_2019 != 'No weather file')]

    df_has_weather = df_has_weather[['nhgis_2010_county_gisjoin', 'nhgis_2010_puma_gisjoin']]
    df_has_weather = df_has_weather.set_index('nhgis_2010_puma_gisjoin')
    weather_lookup_df = df_has_weather[~df_has_weather.index.duplicated()] # remove duplicated indices

    all_counties = set()
    for i in trange(len(bldgs_df)):

        by = building_years[ bldgs_df['building_years'].iloc[i] ]

        weather_path = time_series_dir / by / 'weather'

        puma =  bldgs_df['puma'].iloc[i]
        county = weather_lookup_df.loc[puma]['nhgis_2010_county_gisjoin']
        county_weather_file = weather_path / f'{county}.csv'

        all_counties.add(str(county_weather_file))
    
    print(f'# unique counties in {args.energyplus_index_file}: {len(all_counties)}')


    for c in tqdm(all_counties):
        df = pd.read_csv(c)
        df.columns = weather_columns
        # append to weather_df
        #weather_df = pd.concat([weather_df, df], ignore_index=True)
        for wc in weather_columns[1:]:
            weather_data[wc] += [ df[wc].to_numpy() ]

    for col in weather_columns[1:]:
        print('Fitting StandardScaler...', col)
        ss = StandardScalerTransform()
        w = np.vstack(weather_data[col])
        print(f'Mean {col}: {np.mean(w)}')
        ss.train(w)
        outdir = output_dir / col
        if not outdir.exists():
            os.makedirs(outdir)
        ss.save(outdir)
        print('StandardScaler: ', ss.mean_, ss.std_)

if __name__ == '__main__':
    args = argparse.ArgumentParser()

    args.add_argument('--seed', type=int, default=1, required=False,
                        help='Random seed shuffling. Default: 1')

    args.add_argument('--energyplus_index_file', type=str, 
                      default='comstock_hyperparam_train_seed=42.idx')
    
    args = args.parse_args()

    main(args)