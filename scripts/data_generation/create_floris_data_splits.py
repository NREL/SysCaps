import os
from pathlib import Path
import pandas as pd
import argparse
import numpy as np
import random 


# 60/20/20 split - https://github.com/NREL/windAI_bench/blob/main/FLORIS_PLayGen/baseline_models_wind_plant_data_h5.ipynb
# remove the 2500 optimal_yaw scenarios
# index files just need layout_id and scenario_id

if __name__ == '__main__':

    args = argparse.ArgumentParser()

    args.add_argument('--seed', type=int, default=1, required=False,
                        help='Random seed for shuffling. Default: 1')
    
    args = args.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)


    SYSCAPS_PATH = os.environ.get('SYSCAPS', '')
    if SYSCAPS_PATH == '':
        raise ValueError('SYSCAPS environment variable not set')
    SYSCAPS_PATH = Path(SYSCAPS_PATH)

    opt_yaw_list = pd.read_csv(
        SYSCAPS_PATH / 'metadata' / 'floris' / 'opt_yaw_list.csv', 
        names=['layout_id', 'scenario_id'])
    
    train_layouts, train_scenarios = [], []
    val_layouts, val_scenarios = [], []
    test_layouts, test_scenarios = [], []
    

    # split layout ids 60/20/20 train/val/test
    all_layout_ids = [f'Layout{i:03}' for i in range(500)]
    random.shuffle(all_layout_ids)
    train_layout_ids = all_layout_ids[:300]
    val_layout_ids = all_layout_ids[300:400]
    test_layout_ids = all_layout_ids[400:]

    
    # remove the 2500 optimal_yaw scenarios
    for i in range(500):
        for j in range(500):
            layout = f'Layout{i:03}'
            scenario = f'Scenario{j:03}'
            # check that no row of opt_yaw_list contains this layout and scenario
            if not ((opt_yaw_list['layout_id'] == layout) & (opt_yaw_list['scenario_id'] == scenario)).any():
                #layout_ids.append(str(layout))
                #scenario_ids.append(str(scenario))
                if layout in train_layout_ids:
                    train_layouts.append(layout)
                    train_scenarios.append(scenario)
                elif layout in val_layout_ids:
                    val_layouts.append(layout)
                    val_scenarios.append(scenario)
                else:
                    test_layouts.append(layout)
                    test_scenarios.append(scenario)
    total_datapoints = len(train_layouts)+len(val_layouts)+len(test_layouts)
    assert total_datapoints == (250000 - 2500), f'total_datapoints: {total_datapoints}'

    # write the index files
    for layout_split, scenario_split, split_name in zip([train_layouts, val_layouts, test_layouts],
                                        [train_scenarios, val_scenarios, test_scenarios],
                                        ['train', 'val', 'test']):
        # shuffle the order of the layouts+scenarios within each split
        shuf = list(range(len(layout_split)))
        random.shuffle(shuf)
        layout_split = [layout_split[i] for i in shuf]
        scenario_split = [scenario_split[i] for i in shuf]
        
        fname = f'floris_{split_name}_seed={args.seed}.idx'
        print(f'Creating index file {fname}...')
        idx_file = open(str(SYSCAPS_PATH / 'metadata' / 'splits' / fname), 'w')

        for i in range(len(layout_split)):
            linestr = f'{layout_split[i]}\t{scenario_split[i]}\n'
            idx_file.write(linestr)
        idx_file.close()