# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from typing import Optional
import fire
from llama import Llama
import pandas as pd
import os
from pathlib import Path
import time
from datetime import date, datetime, timedelta, time as t

def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_batch_size: int = 8,
    max_gen_len: Optional[int] = None,
    prompt_length: str = 'medium',
):
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )
    SYSCAPS_PATH = os.environ.get('SYSCAPS', '')
    if SYSCAPS_PATH == '':
        raise ValueError('SYSCAPS environment variable not set')
    SYSCAPS_PATH = Path(SYSCAPS_PATH)

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
    prompts = {
        "short": "Write a building description based on the following attributes. Your answer should be 1-3 sentences. Please note that your response should NOT be a list of attributes and should be entirely based on the information provided. Be as concise as possible.",
        "medium": "Write a building description based on the following attributes. Your answer should be 4-6 sentences. Please note that your response should NOT be a list of attributes and should be entirely based on the information provided.",
        "long": "Write a building description based on the following attributes. Your answer should be 7-9 sentences. Please note that your response should NOT be a list of attributes and should be entirely based on the information provided. Be as detailed as possible."
    }

    savedir_ = SYSCAPS_PATH / 'captions' / 'comstock'
    if not savedir_.exists():
        os.makedirs(savedir_)
    if prompt_length == 'all':
        prompt_lengths = ['short', 'medium', 'long']
    else:
        prompt_lengths = [prompt_length]

    # convert a df row (series) to key, value pairs in string
    def toString(df_row):
        ans = {}
        for k in df_row.keys():              
            v = df_row[k]

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
            ans['in.weekday_operating_hours'] = str(hours.seconds // 3600) + \
                " hours " + str((hours.seconds//60)%60) + " minutes"

        if 'in.weekend_operating_hours' in ans:
            hours = ans['in.weekend_operating_hours']
            ans['in.weekend_operating_hours'] = str(hours.seconds // 3600) + \
                " hours " + str((hours.seconds//60)%60) + " minutes"

        for k in ans:
            if type(ans[k]) == t:
                v = ans[k]
                ans[k] = "%d:%02d" % (v.hour, v.minute)
                ans[k] += " AM" if v.hour < 12 else " PM"
        return ", ".join([k + ": " + str(ans[k]) for k in ans if "operating" not in k])

    metadata_dir = SYSCAPS_PATH / 'Buildings-900K' / 'end-use-load-profiles-for-us-building-stock' / '2021'
    df1 = pd.read_parquet(metadata_dir / 'comstock_amy2018_release_1' / 'metadata' / 'metadata.parquet', engine="pyarrow")
    df2 = pd.read_parquet(metadata_dir / 'comstock_tmy3_release_1' / 'metadata' / 'metadata.parquet', engine="pyarrow")
    df = df1.loc[ df1.index.intersection(df2.index).values ]

    for prompt_length in prompt_lengths:
        save_dir_len = savedir_ / f'{prompt_length}'
        captions = pd.DataFrame(columns=['building_id', 'caption'])
        
        if not save_dir_len.exists():
            os.makedirs(save_dir_len)

        for row_idx,bd_id in enumerate(df.index):
            tic = time.perf_counter()
            
            pt = f'{prompts[prompt_length]}\n' + \
                        toString(df.loc[bd_id][com_names])
            dialogs = [[
            {
                "role": "system",
                "content": "You are a building energy expert who provides building descriptions with an objective tone."
            },
            {
                "role": "user",
                "content": pt
            }]]

            results = generator.chat_completion(
                dialogs,  # type: ignore
                max_gen_len=max_gen_len,
                temperature=temperature,
                top_p=top_p,
            )
            toc = time.perf_counter()
            for dialog, result in zip(dialogs, results):

                for msg in dialog:
                    print(f"{msg['role'].capitalize()}: {msg['content']}\n")
                print(
                    f"> {result['generation']['role'].capitalize()}: {result['generation']['content']}"
                )
                print(f"time = {toc - tic:0.4f} seconds \n")
                print("\n==================================\n")

                # write to files
                #file = open(save_dir_len / f'{bd_id}_cap.txt', 'w')
                #file.write(result['generation']['content'])
                #file.close()
                
                captions.loc[row_idx] = [bd_id, result['generation']['content']]
        # save dataframe to csv
        captions.to_csv(save_dir_len / 'captions.csv', index=False)
       
if __name__ == "__main__":
    fire.Fire(main)