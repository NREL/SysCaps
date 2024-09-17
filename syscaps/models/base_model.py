import abc
from typing import Tuple, Dict, Union
from pathlib import Path
import torch
import torch.nn as nn


class BaseSurrogateModel(nn.Module, metaclass=abc.ABCMeta):
    """Base surrogate model"""
    def __init__(self, is_autoregressive: bool):
        """Init method for BaseSurrogateModel.
        """
        super().__init__()
        self.is_autoregressive = is_autoregressive
    
    @abc.abstractmethod
    def forward(self, x: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass. 
        
        Args:
            x (Dict): dictionary of input tensors
        Returns:
            predictions, distribution parameters (Tuple[torch.Tensor, torch.Tensor]): outputs
        """
        raise NotImplementedError()
    

    @abc.abstractmethod
    def loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """A function for computing the loss.
        
        Args:
            x (torch.Tensor): preds of shape (batch_size, seq_len, 1)
            y (torch.Tensor): targets of shape (batch_size, seq_len, 1)
        Returns:
            loss (torch.Tensor): scalar loss
        """
        raise NotImplementedError()


    @abc.abstractmethod 
    def predict(self, x: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """A function for making a forecast on x with the model.

        Args:
            x (Dict): dictionary of input tensors
        Returns:
            predictions (torch.Tensor): of shape (batch_size, pred_len, 1)
            distribution_parameters (torch.Tensor): of shape (batch_size, pred_len, -1)
        """
        raise NotImplementedError()

    # added device param so that the model can be loaded on cpu
    def load_from_checkpoint(self, checkpoint_path: Union[str, Path], device=torch.device("cuda")):
        """Describes how to load the model from checkpoint_path."""
        stored_ckpt = torch.load(checkpoint_path, map_location=device)
        model_state_dict = stored_ckpt['model']
        new_state_dict = {}
        for k,v in model_state_dict.items():
            # remove string 'module.' from the key
            if 'module.' in k:
                new_state_dict[k.replace('module.', '')] = v
            else:
                new_state_dict[k] = v
        self.load_state_dict(new_state_dict) 