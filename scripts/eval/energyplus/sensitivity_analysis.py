# template is just varying two attributes. target value is annual stock energy consumption.
# output should be a 3D surface plot with two attributes on the x and y axes and the target value on the z axis,
# as well as csv file with the same data.
from syscaps.models import model_factory
from syscaps.data.energyplus import EnergyPlusDataset
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
from sklearn.preprocessing import OneHotEncoder

import os
import pandas as pd
from pathlib import Path 
from datetime import datetime, timedelta

SCRIPT_PATH = Path(os.path.realpath(__file__)).parent
SYSCAPS_PATH = os.environ.get('SYSCAPS', '')
if SYSCAPS_PATH == '':
    raise ValueError('SYSCAPS environment variable not set')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, default='', required=True)
    parser.add_argument('--model_fname', type=str, required=True, help="model file name")
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--ckpt_dir', type=str, required=True, 
                        help="directory of saved checkpoints") 
    
    
    parser.add_argument('--index_files', type=str, default="all", 
                        help="index files, seperated by \",\", default = all")
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--caption_split', type=str, default='medium')
       
    parser.add_argument('--interpolate_attribute_1', action='store_true', default=False)
    #parser.add_argument('--attribute_1', type=str, default='')
    #parser.add_argument('--attribute_2', type=str, default='')
    
    args = parser.parse_args()

    if args.index_files == "all":
        args.index_files = ["comstock_buildings900k_test_seed=42.idx", "comstock_attribute_combos_seed=42.idx"]
    else:
        args.index_files = [idx_file.strip() for idx_file in args.index_files.split(",") if idx_file != ""]

    device     = args.device
    batch_size = args.batch_size
    caption_split = args.caption_split
    model_fname   = args.model_fname

    attribute_1 = 'in.sqft'
    attribute_2 = 'in.number_of_stories'

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
    ckpt_dir = Path(args.ckpt_dir)
    model.load_from_checkpoint(ckpt_dir / model_fname, device=device)

    result_dir = args.results_dir 
    
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)

    template = "The in.number_of_stories in.building_type has a total area of in.sqft square feet. " \
               "It is equipped with a in.hvac_system_type system. "\
               "The building opens at in.weekday_opening_time on weekdays and in.weekend_opening_time on weekends, " \
               "and closes at in.weekday_operating_hours on weekdays and in.weekend_operating_hours on weekends."

    # get the attribute values for the two attributes
    # one-hot encode attributes
    df1 = pd.read_parquet(SYSCAPS_PATH / 'metadata' / "comstock_amy2018.parquet", engine="pyarrow")
    df2 = pd.read_parquet(SYSCAPS_PATH / 'metadata' / "comstock_tmy3.parquet", engine="pyarrow")
    df = df1.loc[ df1.index.intersection(df2.index).values ]
    attributes = open(SYSCAPS_PATH / 'metadata'  / 'attributes_comstock.txt', 'r').read().strip().split('\n')
    attributes = [x.strip('"') for x in attributes]
    attribute_onehot_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
    attribute_onehot_encoder.fit(df[attributes].values)
    values_for_attribute_1 = attribute_onehot_encoder.categories_[attributes.index(attribute_1)]
    values_for_attribute_2 = attribute_onehot_encoder.categories_[attributes.index(attribute_2)]

    if args.interpolate_attribute_1:
        #### !!!!! OOD values for in.sqft
        new_values_for_attribute_1 = list(values_for_attribute_1)
        for i in range(1,len(values_for_attribute_1)):
            new_values_for_attribute_1.append( (values_for_attribute_1[i] + values_for_attribute_1[i-1]) / 2.0)
        values_for_attribute_1 = new_values_for_attribute_1

    print(f'{len(values_for_attribute_1)} values for {attribute_1}: {values_for_attribute_1}')
    print(f'{len(values_for_attribute_2)} values for {attribute_2}: {values_for_attribute_2}')


    df = pd.DataFrame(columns=[attribute_1, attribute_2, 'total_energy_consumption (GWh)'])
    #df = pd.DataFrame(columns=[attribute_1, 'total_energy_consumption (GWh)'])

    # create a grid of attribute values
    row_idx = 0
    for i in range(len(values_for_attribute_1)):
        for j in range(len(values_for_attribute_2)):
            
            total_energy_consumption = 0

            for index_file in args.index_files:
                # given a building, plot actual vs. predicted load
                dataset = EnergyPlusDataset(
                    buildings_bench_path=Path(SYSCAPS_PATH),
                    index_file=index_file,
                    resstock_comstock='comstock',
                    syscaps_split=caption_split,
                    return_full_year=True,
                    include_text = True,
                    tokenizer = model_args["text_encoder_name"]
                )
                template_ij = template.replace(attribute_1, 
                                    dataset.process_attributes_to_caption(attribute_1, values_for_attribute_1[i]))

                template_ij = template_ij.replace(attribute_2, 
                                    dataset.process_attributes_to_caption(attribute_2, values_for_attribute_2[j]))
                dataset.caption_template = template_ij
                print(template_ij)
                
                inverse_normalization_for_qoi_predictions = dataset.load_transform.undo_transform

                dataloader = torch.utils.data.DataLoader(
                    dataset, 
                    batch_size=batch_size,
                    drop_last=False,
                    num_workers = 8,
                    worker_init_fn=utils.worker_init_fn,
                    collate_fn=dataset.collate_fn())  
            
                with torch.no_grad():
                    for batch in tqdm(dataloader):   
                        
                        for k,v in batch.items():
                            if torch.is_tensor(v):
                                batch[k] = v.to(device)

                        with torch.cuda.amp.autocast():
                            preds = model.predict(batch)
                        
                        preds = preds.to(torch.float32)
                        # unscale loads 
                        preds = inverse_normalization_for_qoi_predictions(preds)
                        
                        total_energy_consumption += preds.sum().item()
            total_energy_consumption /= 100000.0 # convert to GWh
            df.loc[row_idx] = {attribute_1: values_for_attribute_1[i], 
                            attribute_2: values_for_attribute_2[j], 
                            'total_energy_consumption (GWh)': total_energy_consumption}
            row_idx += 1
    # attr 1 attr 2
            
    # save to csv
    if args.interpolate_attribute_1:
        df.to_csv(result_dir / f'{args.model_fname}_{attribute_1}_interpolated_{attribute_2}_sensitivity_analysis.csv', index=False)
    else:
        df.to_csv(result_dir / f'{args.model_fname}_{attribute_1}_{attribute_2}_sensitivity_analysis.csv', index=False)
    
    #3d surface plot
    #fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')

    # ax.plot_trisurf(df[attribute_1], df[attribute_2], df['total_energy_consumption (GWh)'], cmap='viridis')
    # ax.set_xlabel(attribute_1)
    # ax.set_ylabel(attribute_2)
    # ax.set_zlabel('GWh')
    # plt.tight_layout()
    # plt.savefig(result_dir / f'{attribute_1}_{attribute_2}_sensitivity_analysis.png')
            
    # ood in.sqft
    # df.to_csv(result_dir / f'{attribute_1}_ood_sensitivity_analysis.csv', index=False)
    # # 2d plot of in.sqft vs total_energy_consumption

    # plt.plot(df[attribute_1], df['total_energy_consumption (GWh)'], marker='o')
    # plt.xlabel(attribute_1)
    # plt.ylabel('GWh')
    # plt.tight_layout()
    # plt.savefig(result_dir / f'{attribute_1}_ood_sensitivity_analysis.png')