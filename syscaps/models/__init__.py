import importlib
import torch
from typing import Dict


def model_factory(module_name: str, dataset: str, model_args: Dict) -> torch.nn.Module:
    """Instantiate and returns a model for the benchmark.

    Returns the model itself.

    Args:
        module_name (str): Name of the module.
        dataset (str): Dataset name (determines model class variant)
        model_args (Dict): The keyword arguments for the model.
    Returns:
        model (torch.nn.Module): the instantiated model  
    """
    try:
        model_mod = importlib.import_module(f"syscaps.models.{module_name}")
    except:
        raise ValueError(f"Module: syscaps.models.{module_name} not found")
    model = model_mod.build(dataset, **model_args)
    return model


