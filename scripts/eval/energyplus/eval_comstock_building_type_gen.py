from syscaps.evaluation import metrics
from syscaps.evaluation.managers import MetricsManager
from syscaps.models import model_factory
from syscaps.data.energyplus import EnergyPlusDataset
from syscaps import utils
import torch 
from tqdm import tqdm
import tomli
import numpy as np
import argparse
from pathlib import Path 
import os
import wandb

## Env variables
SCRIPT_PATH = Path(os.path.realpath(__file__)).parent
SYSCAPS_PATH = os.environ.get('SYSCAPS', '')
if SYSCAPS_PATH == '':
    raise ValueError('SYSCAPS environment variable not set')
SYSCAPS_PATH = Path(SYSCAPS_PATH)

# eval one model on one dataset 
def evaluate(model, dataloader, caption_split, args):
    # building_types = {
    #     'FullServiceRestaurant': 'FineDiningRestaurant',
    #     'RetailStripmall': 'ShoppingCenter',
    #     'Warehouse': 'Big Box Store',
    #     'RetailStandalone': 'ConvenienceStore',
    #     'SmallOffice': 'Co-WorkingSpace',
    #     'PrimarySchool': 'ElementarySchool',
    #     'MediumOffice': 'Workplace',
    #     'SecondarySchool': 'HighSchool',
    #     'Outpatient': 'MedicalClinic',
    #     'QuickServiceRestaurant': 'FastFoodRestaurant',
    #     'LargeOffice': 'OfficeTower',
    #     'LargeHotel': 'Five-Star Hotel',
    #     'SmallHotel': ['Bed and Breakfast', 'Motel'],
    #     'Hospital': 'HealthcareFacility'
    # }

    building_types = ['FullServiceRestaurant','RetailStripmall','Warehouse', 'RetailStandalone',
                   'SmallOffice', 'PrimarySchool', 'MediumOffice', 'SecondarySchool',
                   'Outpatient', 'QuickServiceRestaurant', 'LargeOffice', 'LargeHotel',
                   'SmallHotel', 'Hospital']
        
    model.eval()

    metrics_managers = {}
    # estimate results for each building type separately
    for bt in building_types:
        metrics_list = [metrics.ErrorMetric('nrmse-per-run-hourly',  'simrun', 'hourly', metrics.squared_error, normalize=True, sqrt=True),
                    metrics.ErrorMetric('nmbe-per-run-hourly', 'simrun', 'hourly', metrics.bias_error, normalize=True, sqrt=False)]
        metrics_manager = MetricsManager(
                metrics=metrics_list
        )
        metrics_managers[bt] = metrics_manager 

    with torch.no_grad():
        for batch in tqdm(dataloader):   
            
            onehot = batch['attributes_onehot']
            onehot_np = onehot.numpy()
            attrs = dataloader.dataset.attribute_onehot_encoder.inverse_transform(onehot_np[:,0])
            batch_building_types = attrs[:,1]
            
            for k,v in batch.items():
                if torch.is_tensor(v):
                    batch[k] = v.to(args.device)

            targets = batch['y']

            with torch.cuda.amp.autocast():
                preds = model.predict(batch)

            targets = targets.to(torch.float32)
            preds = preds.to(torch.float32)

            # unscale loads 
            targets = inverse_normalization_for_qoi_predictions(targets)
            preds = inverse_normalization_for_qoi_predictions(preds)

            for i in range(targets.shape[0]):
                metrics_managers[batch_building_types[i]](
                    y_true=targets[i].unsqueeze(0),
                    y_pred=preds[i].unsqueeze(0),
                )

    return metrics_managers

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_name', type=str, required=True,
                        help='Use the same eval_name each time you execute '
                        'this script, for each model you want to compare in '
                         'the same wandb table.')
    parser.add_argument('--model_fnames', type=str, required=True, 
                        help="model file names seperated by \",\"")
    parser.add_argument("--ckpt_dir", type=str, required=True, 
                        help="directory of saved checkpoints")
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--index_files', type=str, default="all", 
                        help="index files, seperated by \",\", default = all")
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--caption_splits', type=str, default='onehot',
                        help='caption splits, separate by \",\", default = onehot')    
    #parser.add_argument('--wandb_project', type=str, default='attribute-caps-comstock-eval',
    #                    help='wandb project name for these eval runs')
    args = parser.parse_args()

    if args.index_files == "all":
        args.index_files = ["comstock_buildings900k_test_seed=42.idx", "comstock_attribute_combos_seed=42.idx"]
    else:
        args.index_files = [idx_file.strip() for idx_file in args.index_files.split(",") if idx_file != ""]

    args.model_fnames = [model_fname.strip() for model_fname in args.model_fnames.split(",") if model_fname != ""]

    args.caption_splits = [caption_split.strip() for caption_split in args.caption_splits.split(",") if caption_split != ""]

    print("index files", args.index_files)
    print("model files", args.model_fnames)
    print("caption splits", args.caption_splits)

    config_path =  SCRIPT_PATH / '..' / '..' / '..' / \
        'syscaps' / 'configs' / 'energyplus_comstock' 
    if (config_path / f'{args.model}.toml').exists():
        toml_args = tomli.load(( config_path / f'{args.model}.toml' ).open('rb'))
    else:
        raise ValueError()

    # grab the custom model args as set in the config file
    model_args = toml_args['model']
    model = model_factory(toml_args['experiment']['module_name'], 'energyplus_comstock', model_args)
    model = model.to(args.device)

    # checkpoint directory
    ckpt_dir = Path(args.ckpt_dir)

    run = wandb.init(
        project='comstock-eval',
        notes=args.model,
        config={**model_args, **vars(args)})

    column_names = ['model', 'dataset', 'caption']
    row_data = []
    i = 0
    for caption_split in args.caption_splits:
        for model_fname in args.model_fnames:
            model.load_from_checkpoint(ckpt_dir / model_fname, device=args.device)

            for index_file in args.index_files:
                print(f"model fname = {model_fname}")
                print(f"index file = {index_file}")
                print(f"caption split = {caption_split}")

                dataset = EnergyPlusDataset(
                    buildings_bench_path=SYSCAPS_PATH,
                    index_file=index_file,
                    resstock_comstock='comstock',
                    syscaps_split=caption_split if caption_split != 'onehot' else 'keyvalue',
                    return_full_year=True,
                    include_text = True
                )
                inverse_normalization_for_qoi_predictions = dataset.load_transform.undo_transform

                dataloader = torch.utils.data.DataLoader(
                    dataset, batch_size=args.batch_size,
                    drop_last=False,
                    num_workers = 8,
                    worker_init_fn=utils.worker_init_fn,
                    collate_fn=dataset.collate_fn())

                metrics_managers = evaluate(model, dataloader, caption_split, args)

                data = [model_fname, index_file, caption_split]
                for building_type_name,v in metrics_managers.items():
                    summary = v.summary()
                    if len(summary) == 0:
                        data += [None,None]
                        if i == 0:
                            column_names += [f'nrmse-per-run-hourly-{building_type_name}']
                            column_names += [f'nmbe-per-run-hourly-{building_type_name}']

                    for kk,vv in summary.items():
                        if kk == 'loss':
                            print(f'loss: {summary[kk]}')
                            data += [summary[kk]]
                        else:
                            print(f'{summary[kk].name}-{building_type_name}: {summary[kk].value}')
                            data += [summary[kk].value]
                        
                        if i == 0:
                            column_names += [kk + f'-{building_type_name}']
                
                i += 1
                print()

                row_data += [data]
    table = wandb.Table(columns=column_names, data=row_data)
    wandb.log({args.eval_name: table})

    run.finish()