import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import random
from tqdm import trange 
import os
from syscaps.transforms import BoxCoxTransform 


def main(args):
    """
    For each building in hyperparam_train...
    """
    random.seed(args.seed)
    np.random.seed(args.seed)

    project_dir = Path(os.environ.get('SYSCAPS', ''))
    output_dir = project_dir / 'metadata' / 'transforms' / args.energyplus_index_file.split('_')[0] / 'load'
    if not output_dir.exists():
        output_dir.mkdir(parents=True)  
    buildings_for_transform = project_dir / 'metadata' / 'splits' / args.energyplus_index_file 
    print(f'Fitting load transform for {str(buildings_for_transform)}')

    # training set dir
    time_series_dir = Path(os.environ.get('SYSCAPS', ''), 'Buildings-900K', 'end-use-load-profiles-for-us-building-stock', '2021')
    building_years = ['comstock_tmy3_release_1', 'resstock_tmy3_release_1', 'comstock_amy2018_release_1', 'resstock_amy2018_release_1'] 
    census_regions = ['by_puma_midwest', 'by_puma_south', 'by_puma_northeast', 'by_puma_west']

    bldgs_df = pd.read_csv(buildings_for_transform, 
                           sep='\t', names=['building_years', 'census', 'puma', 'bldg_id'])

    loads = []
    for i in trange(len(bldgs_df)):
        by = building_years[ bldgs_df['building_years'].iloc[i] ]

        by_path = time_series_dir / by / 'timeseries_individual_buildings'
        
        census = census_regions[ bldgs_df['census'].iloc[i] ]
        puma =  bldgs_df['puma'].iloc[i]
        bldg_id = str(bldgs_df['bldg_id'].iloc[i])

        df = pq.read_table(str(by_path / census / 'upgrade=0' / f'puma={puma}'),
                       columns=['timestamp', bldg_id])
        
        df = df.to_pandas().sort_values(by='timestamp')
       
        # convert each column to a numpy array and stack vertically
        loads += [df[bldg_id].to_numpy()]

    print('Fitting BoxCox...')
    l = np.vstack(loads)
    print(f'Mean power consumption (kWh): {np.mean(l)}')
    bc = BoxCoxTransform()
    bc.train(l)
    bc.save(output_dir)
    print('BoxCox: ', bc.boxcox.lambdas_)
 

        
if __name__ == '__main__':
    args = argparse.ArgumentParser()

    args.add_argument('--seed', type=int, default=1, required=False,
                        help='Random seed shuffling. Default: 1')

    args.add_argument('--energyplus_index_file', type=str, 
                      default='comstock_hyperparam_train_seed=42.idx',
                      )

    args = args.parse_args()

    main(args)
