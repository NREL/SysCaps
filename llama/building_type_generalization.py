# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# load captions, stick in prompt, ask for paraphase in new style? 

from typing import Optional

import fire

from llama import Llama
import pandas as pd
import numpy as np
import os
from pathlib import Path
import time
from datetime import date, datetime, timedelta, time as t

"""
CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node 1 \
    --rdzv-backend=c10d \
    --rdzv-endpoint=localhost:0 \
    building_type_generalization.py \
    --ckpt_dir llama-2-7b-chat/ \
    --tokenizer_path tokenizer.model \
    --max_seq_len 512 --max_batch_size 4 \
    --split_name medium

"""

def main(
    #csv_file: str, # name of the csv file that contains building attributes
    split_name: str, # name of the dataset split (short, medium, long)
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_batch_size: int = 8,
    max_gen_len: Optional[int] = None,
    #worker_id: int = 1, # worker id
    #worker_num: int = 1, # total number of workers
    #debug: bool = False # debug mode, only generate a small set of data (10 samples)
):

    building_types = {
        'FullServiceRestaurant': 'FineDiningRestaurant',
        'RetailStripmall': 'ShoppingCenter',
        'Warehouse': 'Storage', #'Big Box Store',
        'RetailStandalone': 'ConvenienceStore',
        'SmallOffice': 'Co-WorkingSpace',
        'PrimarySchool': 'ElementarySchool',
        'MediumOffice': 'Workplace',
        'SecondarySchool': 'HighSchool',
        'Outpatient': 'MedicalClinic',
        'QuickServiceRestaurant': 'FastFoodRestaurant',
        'LargeOffice': 'OfficeTower',
        'LargeHotel': 'Five-Star Hotel',
        'SmallHotel': ['Bed and Breakfast', 'Motel'],
        'Hospital': 'HealthcareFacility'
    }
    building_subtypes = {
        'strip_mall_restaurant20': 'shopping_center_restaurant20',
        'mediumoffice_nodatacenter':  'workplace_nodatacenter',
        'strip_mall_restaurant30': 'shopping_center_restaurant30',
        'strip_mall_restaurant0': 'shopping_center_restaurant0',
        'strip_mall_restaurant10': 'shopping_center_restaurant10',
        'largeoffice_nodatacenter': 'officetower_nodatacenter',
        'strip_mall_restaurant40': 'shopping_center_restaurant40',
        'largeoffice_datacenter': 'officetower_datacenter',
        'mediumoffice_datacenter': 'workplace_datacenter'
    }

    ## Env variables
    SYSCAPS_PATH = os.environ.get('SYSCAPS', '')
if SYSCAPS_PATH == '':
    raise ValueError('SYSCAPS_PATH environment variable not set')


    ATTRIBUTE_CAPS_PATH = os.environ.get('ATTRIBUTE_CAPS', '')
    if ATTRIBUTE_CAPS_PATH == '':
        raise ValueError('ATTRIBUTE_CAPS environment variable not set')
    ATTRIBUTE_CAPS_PATH = Path(ATTRIBUTE_CAPS_PATH)
    
    attributes = open(ATTRIBUTE_CAPS_PATH / 'metadata' / 'attributes_comstock.txt', 'r').read().split('\n')
    attributes = [x.strip('"') for x in attributes]


    # First, keep only buildings in the intersection of amy2018/tmy3
    stock_amy2018 = pd.read_parquet(ATTRIBUTE_CAPS_PATH / 'metadata' / f'comstock_amy2018.parquet', engine="pyarrow")
    stock_amy2018 = stock_amy2018[stock_amy2018["in.building_type"] == 'Warehouse']
    print(stock_amy2018.shape)
    assert split_name in ["short", "medium", "long"]

    split_name = split_name + "_building_type_warehouse"

    prompts = {
        "short_building_type_warehouse": "Write a building description based on the following attributes. Your answer should be 1-3 sentences. Please note that your response should NOT be a list of attributes and should be entirely based on the information provided. Be as concise as possible.",
        "medium_building_type_warehouse": "Write a building description based on the following attributes. Your answer should be 4-6 sentences. Please note that your response should NOT be a list of attributes and should be entirely based on the information provided.",
        "long_building_type_warehouse": "Write a building description based on the following attributes. Your answer should be 7-9 sentences. Please note that your response should NOT be a list of attributes and should be entirely based on the information provided. Be as detailed as possible."
    }

    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    com_names = [
        "in.building_subtype",
        "in.building_type",
        "in.number_of_stories",
        "in.sqft",
        "in.hvac_system_type",
        "in.weekday_operating_hours",
        "in.weekday_opening_time",
        "in.weekend_operating_hours",
        "in.weekend_opening_time",
        "in.tstat_clg_delta_f",
        "in.tstat_clg_sp_f",
        "in.tstat_htg_delta_f",
        "in.tstat_htg_sp_f"
    ]
    
    # convert a df row (series) to key, value pairs in string
    def toString(df_row, building_type_idx=0):
        ans = {}
        for k in df_row.keys():              
            v = df_row[k]

            if "in.building_type" in k:
                if v == 'SmallHotel':
                    # Bed and Breakfast
                    v = building_types[v][building_type_idx]
                    building_type_idx += 1
                else:
                    v = building_types[v]

            if "in.building_subtype" in k:
                if v in building_subtypes:                    
                    v = building_subtypes[v]
            # ignore attributes with nan value
            if str(v) == "nan":
                continue
            # convert time format from hours to hours + minutes
            if "operating_hours" in k or "opening_time" in k:
                m = v * 60
                h = int(m // 60)
                m = int(m % 60)
                if "opening_time" in k:
                    v = t(hour=h, minute=m)
                else:
                    v = timedelta(hours=h, minutes=m)

            # rename thermostat set points to be more interpretable
            elif k in ["in.tstat_clg_delta_f", "in.tstat_clg_sp_f", "in.tstat_htg_delta_f", "in.tstat_htg_sp_f"]:
                thermostat_type = "cooling" if "clg" in k else "heating"
                occupancy_type = "occupied" if "sp" in k else "unoccupied"
                k = " ".join([occupancy_type, thermostat_type, "temperature set point"])
                if occupancy_type == "unoccupied":
                    k += " difference from occupied state"
                if v == 999:
                    v = "default"
            ans[str(k)] = v

        if 'in.weekday_opening_time' in ans and 'in.weekday_operating_hours' in ans:
            ans['in.weekday_closing_time'] = (datetime.combine(date.today(), ans['in.weekday_opening_time']) + ans['in.weekday_operating_hours']).time()

        if 'in.weekend_opening_time' in ans and 'in.weekend_operating_hours' in ans:
            ans['in.weekend_closing_time'] = (datetime.combine(date.today(), ans['in.weekend_opening_time']) + ans['in.weekend_operating_hours']).time()

        if 'in.weekday_operating_hours' in ans:
            hours = ans['in.weekday_operating_hours']
            ans['in.weekday_operating_hours'] = str(hours.seconds // 3600) + " hours " + str((hours.seconds//60)%60) + " minutes"

        if 'in.weekend_operating_hours' in ans:
            hours = ans['in.weekend_operating_hours']
            ans['in.weekend_operating_hours'] = str(hours.seconds // 3600) + " hours " + str((hours.seconds//60)%60) + " minutes"

        for k in ans:
            if type(ans[k]) == t:
                v = ans[k]
                ans[k] = "%d:%02d" % (v.hour, v.minute)
                ans[k] += " AM" if v.hour < 12 else " PM"
        return ", ".join([k + ": " + str(ans[k]) for k in ans if "operating" not in k]), building_type_idx

    names = com_names
    caption_dir = ATTRIBUTE_CAPS_PATH / "captions" / "comstock"

    #df = pd.read_csv(caption_dir / csv_file)

    # test buildings
    idx_files = ['comstock_buildings900k_test_seed=42.idx', 
                 'comstock_attribute_combos_seed=42.idx']
    

    for idxf in idx_files:

        savedir_ = caption_dir / split_name
        savedir_tokens = caption_dir / (split_name + "_tokens")
        
        if not savedir_.exists():
            os.makedirs(savedir_)
        if not savedir_tokens.exists():
            os.makedirs(savedir_tokens)

        df = pd.read_csv(ATTRIBUTE_CAPS_PATH / 'metadata' / 'splits' / idxf, sep='\t', names=['building_type_year', 'census_region', 'puma', 'bldg_id'])

        for i, row in df.iterrows():
            bd_id = row["bldg_id"]

            try:
                row = stock_amy2018.loc[bd_id]
            except KeyError:
                continue
            
            # if i >= 10 and debug:
            #     break

            # if i % worker_num != worker_id - 1:
            #     continue

            # if os.path.exists(caption_dir / split_name / f"{bd_id}_cap.txt"):
            #     continue

            
            tic = time.perf_counter()
            print(names)
            prompt_string, building_type_idx = toString(row[names], 0)

            if building_type_idx == 1:
                second_prompt_string, _ = toString(row[names], 1)
                p_strings = [prompt_string, second_prompt_string]
            else:
                p_strings = [prompt_string]

            for small_hotel_idx,p_string in enumerate(p_strings):
                prompt = f"{prompts[split_name]} \n" + p_string

                # prompt = ex1 + ex2 + "Building Attributes: "
                # prompt += toString(row[names]) + "\n"
                # prompt += "Building Description: " + "\n"

                dialogs = [[
                {
                    "role": "system",
                    "content": "You are a building energy expert who provides building descriptions with an objective tone."
                },
                {
                    "role": "user",
                    "content": prompt
                }]]

                results = generator.chat_completion(
                    dialogs,  # type: ignore
                    max_gen_len=max_gen_len,
                    temperature=temperature,
                    top_p=top_p,
                )
                toc = time.perf_counter()
                for dialog, result in zip(dialogs, results):
                    #print(f"job id = {i}, worker id = {worker_id} / {worker_num}")
                    for msg in dialog:
                        print(f"{msg['role'].capitalize()}: {msg['content']}\n")
                    print(
                        f"> {result['generation']['role'].capitalize()}: {result['generation']['content']}"
                    )
                    print(f"time = {toc - tic:0.4f} seconds \n")
                    print("\n==================================\n")

                    # write to files
                    if building_type_idx == 1:
                        if small_hotel_idx == 0:
                            print('saving bed and breakfast...')
                            file = open(savedir_ / f"{bd_id}_cap_bb.txt", 'w')
                            file.write(result['generation']['content'])
                            file.close()
                        else:
                            print('saving motel...')
                            file = open(savedir_ / f"{bd_id}_cap_motel.txt", 'w')
                            file.write(result['generation']['content'])
                            file.close()
                    else:
                        file = open(savedir_ / f"{bd_id}_cap.txt", 'w')
                        file.write(result['generation']['content'])
                        file.close()
                
if __name__ == "__main__":
    fire.Fire(main)
