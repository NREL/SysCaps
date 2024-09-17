import argparse
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import random
import glob 
import pandas as pd
import os
from tqdm import tqdm
import pickle

def split_building_ids(metadata_dir: Path, 
                       resstock_comstock: str = 'comstock',
                       seed: int = 1):
    """
    Create dataset splits

    Returns: pd.Dataframes
        train: buildings used for model training
        val: buildings used for early stopping, overlaps with train (attributes are ~iid with train)
        test: buildings used for model testing, overlaps with train (attiributes are ~iid with train)
        hyperparam_train: subset of buildings used for model hyperparam tuning, *attributes distinct from train* 
        hyperparam_val: shared buildings with hyperparam_train to use for early stopping
        withheld_atts: withheld buildings with novel combinations of attributes
        stock_bldgs_900k_test: withheld buildings in novel PUMAs (semantic OOD split)

    """
    # First, keep only buildings in the intersection of amy2018/tmy3
    stock_amy2018 = pd.read_parquet(metadata_dir / f'{resstock_comstock}_amy2018.parquet', engine="pyarrow")
    stock_tmy3 = pd.read_parquet(metadata_dir / f'{resstock_comstock}_tmy3.parquet', engine="pyarrow")
    bd_ids_amy2018 = stock_amy2018.index
    bd_ids_tmy = stock_tmy3.index

    bd_ids_all = bd_ids_amy2018.intersection(bd_ids_tmy).values
    if resstock_comstock == "resstock":
        #resstock_250k_ids = pd.read_csv(metadata_dir / "resstock_250k.csv")["bldg_id"].values # randomly sampled 250k buildings (include some but not all test buildings)
        #resstock_test_ids = pd.read_csv(metadata_dir / "resstock_test.csv")["bldg_id"].values # all test buildings
        resstock_ids = open(metadata_dir / 'resstock_bldg_ids.txt', 'r').readlines()
        resstock_ids = np.array([int(x) for x in resstock_ids])
        bd_ids_all = bd_ids_all[np.isin(bd_ids_all, resstock_ids)]
    # !!!!!!!
    # depending on the weather year, the same building (with same building ID) may
    # have a different PUMA ID number. 
    # We keep train of stocks for amy2018 and tmy3 separately, thus.
    # !!!!!!!
    train_stock = stock_tmy3.loc[bd_ids_all]
    val_test_stock = stock_amy2018.loc[bd_ids_all]

    # drop any with missing puma/census info
    if resstock_comstock == "comstock":
        mask = ~(train_stock['in.census_region_name'].isnull() | val_test_stock['in.census_region_name'].isnull())
    else:
        # different column name in resstock
        mask = ~(train_stock['in.census_region'].isnull() | val_test_stock['in.census_region'].isnull())
    train_stock = train_stock[ mask ]  
    val_test_stock = val_test_stock[ mask ]
    assert len(train_stock) == len(val_test_stock)

    print(f'Total buildings in {resstock_comstock} shared by AMY2018 and TMY3: {len(bd_ids_all)}...')

    with open(metadata_dir / 'splits' / 'pumas_without_weather.tsv', 'r') as f:
        line = f.readlines()[0]
        withheld_pumas = line.strip('\n').split('\t')[:-1]
        withheld_pumas = [x.split('=')[1] for x in withheld_pumas]
    # remove buildings in PUMAs without weather
    train_stock = train_stock[~(train_stock['in.nhgis_puma_gisjoin'].isin(withheld_pumas))]
    val_test_stock = val_test_stock[~(val_test_stock['in.nhgis_puma_gisjoin'].isin(withheld_pumas))]
    assert len(train_stock) == len(val_test_stock)
    
    print(f'removing weather-less PUMAs {withheld_pumas}, remaining buildings {resstock_comstock}: {len(train_stock)}...')

    # split on buildings in buildings-900K-test 
    with open(metadata_dir / 'splits' / 'buildings_900k_test.tsv', 'r') as f:
        # tab separated file
        line = f.readlines()[0]
        bldgs_900k_test = line.strip('\n').split('\t')
        bldgs_900k_test = [x.split('=')[1] for x in bldgs_900k_test]

    stock_bldgs_900k_test = val_test_stock[(val_test_stock['in.nhgis_puma_gisjoin'].isin(bldgs_900k_test))]
    mask = ~( train_stock['in.nhgis_puma_gisjoin'].isin(bldgs_900k_test) | val_test_stock['in.nhgis_puma_gisjoin'].isin(bldgs_900k_test))
    train_stock = train_stock[ mask ]
    val_test_stock = val_test_stock[ mask ]
    assert len(train_stock) == len(val_test_stock)

    print(f'splitting buildings-900k-test (# {len(stock_bldgs_900k_test)}), '
           f'remaining buildings {resstock_comstock}: {len(train_stock)}...')

    if resstock_comstock == "comstock":
        # remove buildings with withheld attribute combos
        withheld_atts = pd.read_csv(metadata_dir / 'splits' / \
            f'{resstock_comstock}_withheld_attribute_combos.csv', index_col='bldg_id')
        train_stock = train_stock[ ~(train_stock.index.isin(withheld_atts.index)) ]
        val_test_stock = val_test_stock[ ~(val_test_stock.index.isin(withheld_atts.index)) ]
        print(f'splitting on withheld attribute combos (# {len(withheld_atts)}), '
            f'remaining buildings {resstock_comstock}: {len(train_stock)}...')
    else:
        withheld_atts = []

    # tune set - 10K buildings
    # hyperparam_train ~ TMY3
    hyperparam_train = train_stock.sample(n=10000, random_state = seed)
    # hyperparam_val ~ AMY2018
    hyperparam_val = val_test_stock[ val_test_stock.index.isin(hyperparam_train.index) ]
    # remove tune set buildings from train/val/test
    train_stock = train_stock.drop(hyperparam_train.index)
    val_test_stock = val_test_stock.drop(hyperparam_val.index)

    # for early stopping on the hyperparam set, grab 100 out of the 10K
    hyperparam_val = hyperparam_val.sample(n=100, random_state = seed)

    # from amy2018
    val_test_buildings = val_test_stock.sample(n=200 , random_state = seed)
    
    # iid val and test - ~100 buildings from stock each
    val = val_test_buildings.iloc[:100]
    test = val_test_buildings.iloc[100:]

    return train_stock, val, test, hyperparam_train, hyperparam_val, withheld_atts, stock_bldgs_900k_test


def main(args):
    """
    Each line in the index file indicates a building and n 
    <building_type_and_year> <census_region> <puma_id> <building_id>
    
    Example: <0-4> <0-4> G17031 23023 

    """
    random.seed(args.seed)
    np.random.seed(args.seed)

    base_dir = os.environ.get('SYSCAPS', '')
    if base_dir == '':
        raise ValueError('Env variable syscaps is not set')
    metadata_dir = Path(base_dir, 'metadata')
    
    #buildings_bench_dir = os.environ.get('BUILDINGS_BENCH', '')
    #time_series_dir = Path(buildings_bench_dir, 'Buildings-900K', 'end-use-load-profiles-for-us-building-stock', '2021')

    # indexes for the index files
    building_years = {
        'comstock_tmy3_release_1': 0,
        'resstock_tmy3_release_1': 1,
        'comstock_amy2018_release_1': 2,
        'resstock_amy2018_release_1': 3
    }

    census = {
        'Midwest': 0, #'by_puma_midwest',
        'South': 1, #'by_puma_south',
        'Northeast': 2, #'by_puma_northeast',
        'West': 3 #'by_puma_west'
    }
    
    train, val, test, hyperparam_train, hyperparam_val, attribute_combos, buildings900k_test = \
        split_building_ids(metadata_dir, args.resstock_comstock, args.seed)
    all_splits = {
        'train': train,  # tmy3 weather
        'val': val,  # amy2018
        'test': test,  # amy2018
        'hyperparam_train': hyperparam_train,  # tmy3 weather
        'hyperparam_val': hyperparam_val,  # amy2018
        'attribute_combos': attribute_combos,  # building attribute combos are OOD, some are tmy3 weather / amy2018
        'buildings900k_test': buildings900k_test  # amy2018
    }
    
    # for each split, randomly shuffle the rows, ...
    for split_name, split in all_splits.items():
        if split_name == "attribute_combos" and args.resstock_comstock == "resstock":
            continue
        fname = f'{args.resstock_comstock}_{split_name}_seed={args.seed}.idx'
        print(f'Creating index file {fname}...')
        idx_file = open(str(metadata_dir / 'splits' / fname), 'w')
        
        split = split.sample(frac=1, random_state = args.seed)

        # iterate over each building in the split
        for row_id in range(len(split)):

            bldg = split.iloc[row_id]
            if args.resstock_comstock == "comstock":
                census_id = census[ bldg['in.census_region_name'] ]
            else:
                census_id = census[ bldg['in.census_region'] ]
            puma_id = bldg['in.nhgis_puma_gisjoin']
            bldg_id = str(bldg.name)
            bldg_id = bldg_id.zfill(6)

            if 'train' in split_name:
                building_type_and_year = building_years[f'{args.resstock_comstock}_tmy3_release_1']
            else:
                building_type_and_year = building_years[f'{args.resstock_comstock}_amy2018_release_1']

                # BUG: Some of the buildings in attribute_combos are tmy3

            # NB: We don't *need* \n at the end of each line, but it makes it easier to count # of lines for dataloading
            linestr = f'{building_type_and_year}\t{census_id}\t{puma_id}\t{bldg_id}\n'
            assert len(linestr) == 21, f'linestr: {linestr}'

            idx_file.write(linestr)

        idx_file.close()


if __name__ == '__main__':
    args = argparse.ArgumentParser()

    args.add_argument('--seed', type=int, default=1, required=False,
                        help='Random seed for shuffling. Default: 1')

    args.add_argument('--resstock_comstock', type=str, default='comstock')

    args = args.parse_args()

    main(args)
