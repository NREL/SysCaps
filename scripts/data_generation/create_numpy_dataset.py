import os
from pathlib import Path
import torch
from tqdm import tqdm
import torch
import numpy as np
from pathlib import Path
import torch.multiprocessing

from syscaps import utils
from syscaps.data.energyplus_numpy import EnergyPlusNumpy
from syscaps.data.wind import WindDataset


torch.multiprocessing.set_sharing_strategy('file_system')

def main(args):
    utils.set_seed(args.random_seed)

    if args.split == 'all':
        splits = ['train', 'val', 'test']  # Added 'test' split
    else:
        splits = [args.split]

    # check environment variables
    SYSCAPS_PATH = os.environ.get('SYSCAPS', '')
    if SYSCAPS_PATH == '':
        raise ValueError('SYSCAPS environment variable not set')

    for split in splits:
        if split == 'train':
            if args.dataset == 'energyplus_comstock' or args.dataset == 'energyplus_resstock':
                resstock_comstock = 'comstock' if args.dataset == 'energyplus_comstock' else 'resstock'
                train_dataset = EnergyPlusNumpy(
                    data_path = Path(SYSCAPS_PATH),
                    resstock_comstock = resstock_comstock,
                    index_file = args.train_idx_file,
                    return_full_year = False,
                    hours_per_building = args.hours,
                    ordinal_encode = args.ordinal_encode
                )

                train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=10000, 
                                                            worker_init_fn=utils.worker_init_fn, num_workers=70) 
            
            elif args.dataset == 'wind':
                train_dataset = WindDataset(
                    data_path=Path(SYSCAPS_PATH),
                    index_file=args.train_idx_file
                )

                train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=10000, 
                                                            worker_init_fn=utils.worker_init_fn, collate_fn=train_dataset.collate_fn(),
                                                            num_workers=70, shuffle=True) 

            
        elif split == 'val':
            if args.dataset == 'energyplus_comstock' or args.dataset == 'energyplus_resstock':
                resstock_comstock = 'comstock' if args.dataset == 'energyplus_comstock' else 'resstock'
                val_dataset = EnergyPlusNumpy(
                    buildings_bench_path = Path(SYSCAPS_PATH),
                    resstock_comstock = resstock_comstock,
                    index_file = args.val_idx_file,
                    return_full_year = True,
                    hours_per_building = args.hours,
                    ordinal_encode = args.ordinal_encode
                )

                val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=10000, 
                                                            worker_init_fn=utils.worker_init_fn, num_workers=30)
            
            elif args.dataset == 'wind':
                val_dataset = WindDataset(
                    data_path=Path(SYSCAPS_PATH),
                    index_file=args.val_idx_file
                )

                val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=10000, 
                                                            worker_init_fn=utils.worker_init_fn, collate_fn=val_dataset.collate_fn(),
                                                            num_workers=30) 

            
        elif split == 'test':  # Added 'test' split
            if args.dataset == 'energyplus_comstock' or args.dataset == 'energyplus_resstock':
                resstock_comstock = 'comstock' if args.dataset == 'energyplus_comstock' else 'resstock'
                test_dataset = EnergyPlusNumpy(
                    buildings_bench_path = Path(SYSCAPS_PATH),
                    resstock_comstock = resstock_comstock,
                    index_file = args.test_idx_file,  # Added test index file argument
                    return_full_year = True,
                    hours_per_building = args.hours,
                    ordinal_encode = args.ordinal_encode
                )

                test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=10000, 
                                                            worker_init_fn=utils.worker_init_fn, num_workers=30)
                
            elif args.dataset == 'wind':
                test_dataset = WindDataset(
                    data_path=Path(SYSCAPS_PATH),
                    index_file=args.test_idx_file
                )

                test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=10000, 
                                                            worker_init_fn=utils.worker_init_fn, collate_fn=test_dataset.collate_fn(),
                                                            num_workers=30) 
            

    for split in splits:
        if split == 'train':
            print(f'Loading train data with {len(train_dataset)} hours...')
            train_batch = []
            if args.dataset == 'energyplus_comstock' or args.dataset == 'energyplus_resstock':
                for batch in tqdm(train_loader):
                    train_batch.append(batch)
            elif args.dataset == 'wind':
                for batch in tqdm(train_loader):
                    exog= torch.cat([
                            batch["wind_speed"],
                            batch["wind_direction"],
                            batch["turbulence_intensity"]
                        ], dim=1)
                    onehot = batch["attributes_onehot"]
                    onehot = onehot.squeeze()
                    y = batch["y"]
                    batch = torch.cat((exog, onehot, y), dim=1)
                    train_batch.append(batch) 
                
            train_batch = torch.cat(train_batch, dim=0)

        elif split == 'val':
            print(f'Loading validation data with {len(val_dataset)} hours...')
            val_batch = []
            if args.dataset == 'energyplus_comstock' or args.dataset == 'energyplus_resstock':
                for batch in tqdm(val_loader):
                    val_batch.append(batch)
            elif args.dataset == 'wind':
                for batch in tqdm(val_loader):
                    exog= torch.cat([
                            batch["wind_speed"],
                            batch["wind_direction"],
                            batch["turbulence_intensity"]
                        ], dim=1)
                    onehot = batch["attributes_onehot"]
                    onehot = onehot.squeeze()
                    y = batch["y"]
                    batch = torch.cat((exog, onehot, y), dim=1)     
                    val_batch.append(batch) 
                
            val_batch = torch.cat(val_batch, dim=0)
        
        elif split == 'test':  # Added 'test' split
            print(f'Loading test data with {len(test_dataset)} hours...')
            test_batch = []
            if args.dataset == 'energyplus_comstock' or args.dataset == 'energyplus_resstock':
                for batch in tqdm(test_loader):
                    test_batch.append(batch)
            elif args.dataset == 'wind':
                for batch in tqdm(test_loader):
                    exog= torch.cat([
                            batch["wind_speed"],
                            batch["wind_direction"],
                            batch["turbulence_intensity"]
                        ], dim=1)
                    onehot = batch["attributes_onehot"]
                    onehot = onehot.squeeze()
                    y = batch["y"]
                    batch = torch.cat((exog, onehot, y), dim=1)  
                    test_batch.append(batch) 
            
            test_batch = torch.cat(test_batch, dim=0)


    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for split in splits:
        if split == 'train':
            print('Saving train data...')
            X_train = train_batch[:, :-1].numpy()
            y_train = train_batch[:, -1].numpy()

            if args.dataset == 'energyplus_comstock' or args.dataset == 'energyplus_resstock':
                # save numpy arrays
                np.save(output_dir / f'X_train_{args.dataset}_{args.hours}.npy', X_train)
                np.save(output_dir / f'y_train_{args.dataset}_{args.hours}.npy', y_train)
            elif args.dataset == 'wind':
                np.save(output_dir / f'X_train_{args.dataset}.npy', X_train)
                np.save(output_dir / f'y_train_{args.dataset}.npy', y_train)
            
        elif split == 'val':
            print('Saving validation data...')
            X_val = val_batch[:, :-1].numpy()
            y_val = val_batch[:, -1].numpy()
            # save numpy arrays
            np.save(output_dir / f'X_val_{args.dataset}.npy', X_val)
            np.save(output_dir / f'y_val_{args.dataset}.npy', y_val)
        
        elif split == 'test':  # Added 'test' split
            print('Saving test data...')
            X_test = test_batch[:, :-1].numpy()
            y_test = test_batch[:, -1].numpy()
            # save numpy arrays

            if args.test_name == '':
                np.save(output_dir / f'X_test_{args.dataset}.npy', X_test)
                np.save(output_dir / f'y_test_{args.dataset}.npy', y_test)
            else:
                np.save(output_dir / f'X_test_{args.dataset}_{args.test_name}.npy', X_test)
                np.save(output_dir / f'y_test_{args.dataset}_{args.test_name}.npy', y_test)
            

if __name__ == "__main__":
    # parsing arguments
    import argparse
    parser = argparse.ArgumentParser(description='Create numpy dataset for LightGBM')

    parser.add_argument('--random-seed', type=int, default=42)
    parser.add_argument('--hours', type=int, default=438, help='number of hours per building')
    parser.add_argument('-o', '--output-dir', type=str, default='.', help='directory to save dataset')
    parser.add_argument('--dataset', type=str, default='energyplus_comstock', required=True,
                        choices=['energyplus_comstock', 'energyplus_resstock', 'wind'])
    parser.add_argument('--ordinal_encode', default=False, action="store_true")
    parser.add_argument('--split', type=str, default='all', choices=['train', 'val', 'test', 'all'], 
                        help='split to create dataset for')
    parser.add_argument('--train_idx_file', type=str, default='',
                        help='Name of index files for training')
    parser.add_argument('--val_idx_file', type=str, default='',
                        help='Name of index files for validation')
    parser.add_argument('--test_idx_file', type=str, default='',
                        help='Name of index files for test')
    parser.add_argument('--test-name', type=str, default='',
                        help='Naming the test split')
        

    args = parser.parse_args()

    main(args)
