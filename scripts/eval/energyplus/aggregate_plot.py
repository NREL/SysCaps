from syscaps.evaluation.managers import MetricsManager
from syscaps.evaluation import metrics
from syscaps.models import model_factory
from syscaps.data.energyplus import EnergyPlusDataset
from syscaps.evaluation.plotting import plot_energyplus
from syscaps.evaluation.plot_utils import TIMESTAMPS, DailyTracker, BuildingsTracker, MetricsTracker
from syscaps import utils

import torch 
from tqdm import tqdm
import tomli
import numpy as np
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1.25)
import pickle
import os
import pandas as pd
from pathlib import Path 


## Env variables
SCRIPT_PATH = Path(os.path.realpath(__file__)).parent
SYSCAPS_PATH = os.environ.get('SYSCAPS', '')
if SYSCAPS_PATH == '':
    raise ValueError('SYSCAPS_PATH environment variable not set')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_fname', type=str, required=True, help="model file name")
    parser.add_argument('--model', type=str, required=True,
                        help="{onehot/keyvalue/medium}/{model name}")
    parser.add_argument('--ckpt_dir', type=str, required=True, 
                        help="directory of saved checkpoints")      
    parser.add_argument('--results_folder', type=str, default='')    
    parser.add_argument('--index_files', type=str, default="all", 
                        help="index files, seperated by \",\", default = all")
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--batch_size', type=int, default=64)  
    parser.add_argument('--caption_split', type=str, default='medium')
    parser.add_argument('--caption_template', type=str, default='')
    parser.add_argument('--resume_from_checkpoint', default=False, action="store_true", help="plot using saved results")
    
    args = parser.parse_args()

    if args.index_files == "all":
        args.index_files = ["comstock_buildings900k_test_seed=42.idx", "comstock_attribute_combos_seed=42.idx"]
    else:
        args.index_files = [idx_file.strip() for idx_file in args.index_files.split(",") if idx_file != ""]

    device     = args.device
    batch_size = args.batch_size
    caption_split = args.caption_split
    model_fname   = args.model_fname


    # cur_dir = Path(__file__).parent.resolve()
    # if not os.path.exists(cur_dir / "plots"):
    #     os.mkdir(cur_dir / "plots")
    # PLOT_PATH = cur_dir / "plots"

    # all hours in year 2018 (excluding 2018-3-11 02:00:00)

    # load a model
    config_path =  SCRIPT_PATH / '..' / '..' / '..' \
        'syscaps' / 'configs' / f'energyplus_comstock' 
    if (config_path / f'{args.model}.toml').exists():
        toml_args = tomli.load(( config_path / f'{args.model}.toml' ).open('rb'))
    else:
        raise ValueError()

    # grab the custom model args as set in the config file
    model_args = toml_args['model']
    model = model_factory(toml_args['experiment']['module_name'], 'energyplus_comstock', model_args)
    model = model.to(device)
    model.eval()
    model.load_from_checkpoint(Path(args.ckpt_dir) / model_fname, device=device)

    metrics_manager = MetricsManager(
            metrics=[metrics.ErrorMetric('nrmse-stock-annual',  'stock', 'annual', metrics.squared_error, normalize=True, sqrt=True),
                    metrics.ErrorMetric('nmbe-stock-annual', 'stock', 'annual', metrics.bias_error, normalize=True, sqrt=False)],
    ) 
    
    df1 = pd.read_parquet(SYSCAPS_PATH / 'metadata' / "comstock_amy2018.parquet", engine="pyarrow")
    df2 = pd.read_parquet(SYSCAPS_PATH / 'metadata' / "comstock_tmy3.parquet", engine="pyarrow")
    
    df = df1.loc[ df1.index.intersection(df2.index).values ]
    groups = {}
    for group in df["in.building_type"].unique():
        groups[group] = df[df["in.building_type"] == group].index.unique()

    result_dir = SCRIPT_PATH / '..' / '..' / '..' / 'results' / 'paper_figures'
    nrmse_results_dir = result_dir 
    if args.results_folder != '':
        result_dir = result_dir / args.results_folder

    if not os.path.exists(result_dir):
        os.mkdir(result_dir)

    # check if template file exists already
    if (result_dir / 'template.txt').exists():
        with open(result_dir / 'template.txt', 'r') as f:
            template = f.read()
            args.caption_template = template

    if not args.resume_from_checkpoint:
        trackers = {
            #"yearly_target": YearlyTracker(groups=groups),
            #"yearly_prediction": YearlyTracker(groups=groups),
            "daily_target": DailyTracker(groups=groups),
            "daily_prediction": DailyTracker(groups=groups),
            "buildings": BuildingsTracker(groups=groups),
            "metrics": MetricsTracker(groups=groups)
        }
        for index_file in args.index_files:
            # given a building, plot actual vs. predicted load
            dataset = EnergyPlusDataset(
                buildings_bench_path=Path(SYSCAPS_PATH),
                index_file=index_file,
                resstock_comstock='comstock',
                syscaps_split=caption_split,
                return_full_year=True,
                include_text = True,
                caption_template=args.caption_template
            )
            inverse_normalization_for_qoi_predictions = dataset.load_transform.undo_transform

            dataloader = torch.utils.data.DataLoader(
                dataset, 
                batch_size=batch_size,
                drop_last=False,
                num_workers = 8,
                worker_init_fn=utils.worker_init_fn,
                collate_fn=dataset.collate_fn())  
        
            with torch.no_grad():
                i = 0
                for batch in tqdm(dataloader):   
                    
                    for k,v in batch.items():
                        if torch.is_tensor(v):
                            batch[k] = v.to(device)

                    targets = batch['y']
                    if i == 0:
                        print(batch['syscaps'][0])
                    
                    with torch.cuda.amp.autocast():
                        preds = model.predict(batch)
                        batch_loss = model.loss(preds, targets)

                    targets = targets.to(torch.float32)
                    preds = preds.to(torch.float32)

                    # unscale loads 
                    targets = inverse_normalization_for_qoi_predictions(targets)
                    preds = inverse_normalization_for_qoi_predictions(preds)

                    metrics_manager(
                        y_true=targets,
                        y_pred=preds,
                        loss=batch_loss
                    )
                            
                    building_ids = np.array(batch["building_id"])
                    
                    #trackers["yearly_target"].update(targets, building_ids)
                    #trackers["yearly_prediction"].update(preds, building_ids)
                    trackers["daily_target"].update(targets, building_ids)
                    trackers["daily_prediction"].update(preds, building_ids)
                    trackers["buildings"].update(None, building_ids)
                    trackers["metrics"].update((targets, preds), building_ids)
                    i += 1

        # save trackers
        with open(result_dir / 'trackers.pkl', 'wb') as f:
            pickle.dump(trackers, f)
   
    # make yearly plots and save them
    # for group in groups:
    #     num_buildings = trackers["buildings"].get(group)
    #     if num_buildings == 0:
    #         continue
    #     nrmse, nmbe = trackers["metrics"].get(group)
    #     plt.figure(figsize=(6,6))
    #     plot_energyplus(
    #         timestamps = TIMESTAMPS,
    #         ground_truth= trackers["yearly_target"].get(group),
    #         prediction = trackers["yearly_prediction"].get(group),
    #         model_name = args.model_type,
    #         title = f'{group} ({num_buildings})\nNRMSE = {nrmse:.4f} NMBE = {nmbe:.4f}',
    #         range = (0, 8759)
    #     )
    #     plt.savefig(result_dir / f"yearly_{group}.png")
    #     plt.close()

    else:
        # load saved trackers
        print("resume from checkpoint")
        with open(result_dir / 'trackers.pkl', 'rb') as f:
            trackers = pickle.load(f)

    summary = metrics_manager.summary(with_loss=True)

    with open(nrmse_results_dir / 'nrmse_results.txt', 'a+') as f:
        f.write(f"{args.results_folder},{summary['nrmse-stock-annual'].value},{summary['nmbe-stock-annual'].value}\n")

    # make daily plots and save them
    for group in groups:
        num_buildings = trackers["buildings"].get(group)
        if num_buildings == 0:
            continue
        nrmse, nmbe = trackers["metrics"].get(group)
        plt.figure(figsize=(8,6))
        plot_energyplus(
            timestamps = np.arange(0, 24),
            ground_truth= trackers["daily_target"].get(group),
            prediction = trackers["daily_prediction"].get(group),
            model_name = 'Prediction',
            title = f'{group} ({num_buildings})',
            range = (0, 23)
        )
        plt.tight_layout()
        plt.savefig(result_dir / f"daily_{group}.png")
        plt.savefig(result_dir / f"daily_{group}.pdf")
        plt.close()
        