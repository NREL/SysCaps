# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from typing import Optional
import fire
from llama import Llama
import pandas as pd
import os
import time
import h5py 
from pathlib import Path


def main(
    wind_data_dir: str,
    output_dir: str,
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_batch_size: int = 8,
    max_gen_len: Optional[int] = None,
    prompt_augmentation: Optional[str] = "with an objective tone",
):
    # Wind plant attributes, fixed for this dataset
    rotor_diameter = 130 # meters
    rated_power = 3.4 # MW

    if prompt_augmentation == "all":
        augment = ['with an objective tone. Creative paraphrasing is acceptable', 
                   'with an objective tone', 
                   'to a colleague', 
                   'to a classroom']
    else:
        augment = [prompt_augmentation]

    prompt =  "Write a wind plant description based on the following attributes. " \
              "Your answer should be 4-6 sentences.  Please note that your response " \
              "should NOT be a list of attributes and should be entirely based on the information provided."

    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    def toString(Layout, layout_type, mrd):
        attrs = ""
        attrs += "Plant layout: " + layout_type + ", "
        attrs += "Number of turbines: " + str(Layout['Number of Turbines'][()]) + ", "
        attrs += "Rotor diameter: " + str(rotor_diameter) + " meters, "
        attrs += "Mean turbine spacing: " + f"{mrd} times the rotor diameter, "
        attrs += "Turbine rated power: " + str(rated_power) + " MW."
        return attrs

    savedir_ = Path(output_dir) / "wind"
    if not savedir_.exists():
        os.makedirs(savedir_)
    
    metadata = pd.read_csv(Path(wind_data_dir) / 'wind_metadata.csv')

    with h5py.File(Path(wind_data_dir) / 'wind_plant_data.h5', 'r') as hf:
        for prompt_aug_idx, prompt_aug in enumerate(augment):
            # if prompt_aug_idx < 3:
            #     continue
            save_dir_aug = savedir_ / f'aug_{prompt_aug_idx}'
            if not save_dir_aug.exists():
                os.makedirs(save_dir_aug)

            layout_names = [k for k in hf.keys() if 'Layout' in k]
            for idx, layout in enumerate(layout_names):
                
                tic = time.perf_counter()

                pt = f"{prompt}\n" + \
                         toString(hf[layout], metadata.iloc[idx]['Layout Type'], metadata.iloc[idx]['Mean Turbine Spacing'])

                dialogs = [[
                {
                    "role": "system",
                    "content": f"You are a wind energy expert describing a wind plant {prompt_aug}."
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
                    #print(f"job id = {i}, worker id = {worker_id} / {worker_num}")
                    for msg in dialog:
                        print(f"{msg['role'].capitalize()}: {msg['content']}\n")
                    print(
                        f"> {result['generation']['role'].capitalize()}: {result['generation']['content']}"
                    )
                    print(f"time = {toc - tic:0.4f} seconds \n")
                    print("\n==================================\n")

                    # write to files
                    file = open(save_dir_aug / f'{layout}_cap.txt', 'w')
                    file.write(result['generation']['content'])
                    file.close()
                    
                
if __name__ == "__main__":
    fire.Fire(main)
