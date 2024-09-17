import fire

from llama import Llama
from pathlib import Path
import os

def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 128,
    max_gen_len: int = 16,
    max_batch_size: int = 4,
):
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    comstock_building_types = 'Full Service Restaurant|Retail Stripmall|Warehouse|Retail Standalone|Small Office|Primary School|Medium Office|Secondary School|Outpatient|Quick Service Restaurant|Large Office|Large Hotel|Small Hotel|Hospital'.split('|')
    comstock_building_subtypes = 'strip_mall_restaurant20|mediumoffice_nodatacenter|strip_mall_restaurant30|strip_mall_restaurant0|strip_mall_restaurant10|largeoffice_nodatacenter|strip_mall_restaurant40|largeoffice_datacenter|mediumoffice_datacenter'.replace('_', ' ').split('|')
    #data_path = Path(os.environ.get('ATTRIBUTE_CAPS',''))

    # Few shot prompt (providing a few examples before asking model to complete more);
    #"""Translate English to French:
    #
    #sea otter => loutre de mer
    #peppermint => menthe poivrée
    #plush girafe => girafe peluche
    #cheese =>""",
    bt_prompt_template = """Return a similar building:

        deli => restaurant
        office => shared workspace
        corner store => bodega
        grocery store => supermarket
        coffee shop => cafe
        BUILDING_TYPE => """
    

    bt_subtype_prompt_template = """Return a similar building subtype:

        department store cafeteria => department store canteen
        medium-rise apartment gym => medium-rise apartment sauna
        high-rise apartment penthouse => high-rise apartment gym
        outlet mall arcade => outlet mall cafeteria
        sports complex lockerroom => sports complex lobby
        BUILDING_SUBTYPE =>"""
    
    all_prompts = []
    for cbt in comstock_building_types:
        all_prompts += [bt_prompt_template.replace('BUILDING_TYPE', cbt)]
    for cbt in comstock_building_subtypes:
        all_prompts += [bt_subtype_prompt_template.replace('BUILDING_SUBTYPE', cbt)]
    
    for p in all_prompts:
        results = generator.text_completion(
            [p] * max_batch_size,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )
        for prompt, result in zip([p]* max_batch_size, results):
            print(prompt)
            print(f"> {result['generation']}")
            print("\n==================================\n")


if __name__ == "__main__":
    fire.Fire(main)