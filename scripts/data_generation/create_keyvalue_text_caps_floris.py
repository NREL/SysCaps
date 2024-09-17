from transformers import DistilBertTokenizer, BertTokenizer
from pathlib import Path
import os 
import numpy as np
import pandas as pd
import h5py


if __name__ == '__main__':
    tok_type = "distilbert-base-uncased"
    if tok_type == "bert-base-uncased":
        tokenizer = BertTokenizer.from_pretrained(tok_type)
    else:
        tokenizer = DistilBertTokenizer.from_pretrained(tok_type)

    SYSCAPS_PATH = os.environ.get('SYSCAPS', '')
    if SYSCAPS_PATH == '':
        raise ValueError('SYSCAPS_PATH environment variable not set')
    SYSCAPS_PATH = Path(SYSCAPS_PATH)

    attributes = open(SYSCAPS_PATH / 'metadata' / 'attributes_floris.txt', 'r').read().split('\n')
    
    savedir = SYSCAPS_PATH / 'captions' / 'floris' / 'basic'
    savedir_tokens = SYSCAPS_PATH / 'captions' / 'floris' / 'basic_tokens' / tok_type

    if not savedir.exists():
        os.makedirs(savedir)
    if not savedir_tokens.exists():
        os.makedirs(savedir_tokens)

    
    data_path =  SYSCAPS_PATH / 'captions' / 'wind_plant_data.h5'
    layout_types = pd.read_csv(SYSCAPS_PATH / 'metadata' / 'floris' / 'layout_type.csv')
    mean_rotor_diameters = pd.read_csv(SYSCAPS_PATH / 'metadata' / 'floris' /'results_mean_turbine_spacing.txt', header=None)

    with h5py.File(data_path, 'r') as hf:
        layout_names = [k for k in hf.keys() if 'Layout' in k]
        for idx, layout in enumerate(layout_names): 
    
            attrs = ""
            attrs += f"Plant Layout:{layout_types.iloc[idx]['Layout Type']}|"
            attrs += f"Number of Turbines:{hf[layout]['Number of Turbines'][()]}|"
            attrs += f"Mean Turbine Spacing:{mean_rotor_diameters.iloc[idx].values[0]}|"
            attrs += "Rotor Diameter:130.0 meters|"
            attrs += "Turbine Rated Power:3.4 MW"

            encoded_captions = tokenizer([attrs], padding=False, truncation=False) 
            

            with open(savedir / f'{layout}_cap.txt', 'w') as f:
                f.write(attrs)
            # save tokenized_captions
            np.save( savedir_tokens / f'{layout}_cap_ids.npy', encoded_captions['input_ids'][0])