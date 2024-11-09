import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import random
import os
from tqdm import trange 
from syscaps.transforms import BoxCoxTransform 
import h5py

def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)

    project_dir = Path(os.environ.get('SYSCAPS', ''))
    output_dir = project_dir / 'metadata' / 'transforms' / 'wind' / 'power'
    wind_for_transform = project_dir / 'metadata' / 'splits' / args.wind_index_file    
    data_path = project_dir / 'captions' / 'wind_plant_data.h5'
    
    print('Fitting power transform...')

    with h5py.File(data_path, 'r') as f:
        df = pd.read_csv(wind_for_transform, sep='\t', names=['layout_id', 'scenario_id'])
        values = []
        for i in trange(len(df)):
            layout_id = df['layout_id'].iloc[i]
            scenario_id = df['scenario_id'].iloc[i]
            w = np.sum(f[layout_id]['Scenarios'][scenario_id]['Turbine Power'][:]) / 1e6
            values += [w]

        #print('Fitting BoxCox...')
        l = np.vstack(values)
        print(f'Mean power consumption (MWh): {np.mean(l)}')
        # max
        print(f'Max power consumption (MWh): {np.max(l)}')
        # min
        print(f'Min power consumption (MWh): {np.min(l)}')

        bc = BoxCoxTransform()
        bc.train(l)
        if not output_dir.exists():
            output_dir.mkdir(parents=True)
        bc.save(output_dir)
        print('BoxCox: ', bc.boxcox.lambdas_)

        # plot values as a histogram
        # import matplotlib.pyplot as plt
        # plt.hist(l, bins=100)
        # plt.title('Power')
        # # save
        # plt.savefig(output_dir / 'power_hist.png')

if __name__ == '__main__':
    args = argparse.ArgumentParser()

    args.add_argument('--seed', type=int, default=1, required=False,
                        help='Random seed shuffling. Default: 1')

    args.add_argument('--wind_index_file', type=str, 
                      default='floris_train_seed=42.idx')
    
    args = args.parse_args()

    main(args)