import torch
from syscaps.models.base_model import BaseSurrogateModel
from syscaps.models.modules import TimeSeriesSinusoidalPeriodicEmbedding
from syscaps.models.modules import ResBlock
from syscaps.models.modules import TextEncoder, OneHotAttributeEncoder, IgnoreOneHotAttributes

import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict


def build(dataset, **kwargs):
    if dataset == 'energyplus_comstock' or dataset == 'energyplus_resstock':
        """Builds a EnergyPlusResNet model."""
        return EnergyPlusResNet(**kwargs)
    elif dataset == 'wind':
        return WindResNet(**kwargs)

class ResNet(BaseSurrogateModel):
    """
    Non-sequential ResNet MLP for learning to map sim inputs to outputs.
    """
    def __init__(self,   
                 attribute_encoder_type: str, # 'onehot' or 'text'
                 resnet_hidden_dim: int,
                 resnet_num_blocks: int,
                 exog_input_dim: int,
                 onehot_attr_input_dim: int,
                 mlp_hidden_dim: int,
                 qoi_dim: int, # dimension of sim output
                 text_encoder_name: str,
                 text_freeze_encoder: bool,
                 text_finetune_only_specific_layers: bool,
                 continuous_head: str,
                 ignore_attributes: bool,
                 is_sequential: bool = False):
        """
        Args:
            attribute_encoder_type (str): 'onehot' or 'text'
            resnet_hidden_dim (int): hidden dim for all residual blocks
            resnet_num_blocks (int): number of residual blocks 
            onehot_attr_input_dim (int): dimension of the onehot attribute input
            mlp_hidden_dim (int): dimension of the hidden layer in the MLP
            qoi_dim (int): dimension of the sim output
            text_encoder_name (str): name of the text encoder
            text_freeze_encoder (bool): whether to freeze the text encoder
            continuous_head (str): 'mse' or 'gaussian_nll'
            ignore_attributes (bool)
            is_sequential (bool)
        """
        super(ResNet,self).__init__(is_sequential)
        self.attribute_encoder_type = attribute_encoder_type 
        
        if attribute_encoder_type == 'onehot':
            # use the same embedding dim as the text encoder would
            attr_output_dim = TextEncoder(
                model_name = text_encoder_name,
                freeze = text_freeze_encoder,
                finetune_only_specific_layers = text_finetune_only_specific_layers
            ).output_dim

            if not ignore_attributes:
                self.attr_encoder = OneHotAttributeEncoder(
                    input_dim= onehot_attr_input_dim,
                    output_dim = attr_output_dim
                )
            else:
                self.attr_encoder = IgnoreOneHotAttributes(
                    input_dim = onehot_attr_input_dim,
                    output_dim = attr_output_dim
                )

        elif attribute_encoder_type == 'text':
            self.attr_encoder = TextEncoder(
                model_name = text_encoder_name,
                freeze = text_freeze_encoder,
                finetune_only_specific_layers = text_finetune_only_specific_layers
            )
            attr_output_dim = self.attr_encoder.output_dim

        resnet_hidden_dims = [resnet_hidden_dim] * resnet_num_blocks
        resnet_input_dim = attr_output_dim + exog_input_dim
        blocks = []
        for block_idx in range(len(resnet_hidden_dims)):
            if block_idx == 0:
                in_dim = resnet_input_dim
                out_dim = resnet_hidden_dims[0]
            else:
                in_dim, out_dim = resnet_hidden_dims[block_idx-1], resnet_hidden_dims[block_idx]
            downsample = None
            if in_dim != out_dim:
                downsample = nn.Linear(in_dim, out_dim)
            block = ResBlock(in_dim=in_dim, out_dim=out_dim, downsample=downsample)
            blocks.append(block)

        self.blocks = nn.Sequential(*blocks)    

        # mlp + linear readout
        self.continuous_head = continuous_head
        out_dim = qoi_dim if self.continuous_head == 'mse' else qoi_dim*2        
        self.mlp = nn.Sequential(
            nn.Linear(resnet_hidden_dims[-1], mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, out_dim)
        )     

    def forward(self, 
                x_exog,
                y_onehot,
                y_text: Dict):
        """
        Can be applied to inputs of shape (B,D) or (B,T,D)

        Args:
            x_exog (torch.Tensor) of shape [batch_size, feature_dim]
            y_onehot (torch.Tensor) of shape [batch_size, 1, onehot_feature_dim]
            y_text (Dict) of with keys 'input_ids' and 'attention_mask', each 
                torch.Tensors   

        Returns:
            (torch.Tensor) of shape [batch_size, qoi_dim] if continuous_head == 'mse',
            (torch.Tensor) of shape [batch_size, 2*qoi_dim] if continuous_head == 'gaussian_nll'
        """        
        if len(x_exog.shape) == 2:
            x_exog = x_exog.unsqueeze(1)
            seq_len = 1
        elif len(x_exog.shape) == 3:
            seq_len = x_exog.shape[1]

        if self.attribute_encoder_type == 'text':
            g = self.attr_encoder(y_text['input_ids'], y_text['attention_mask'])
            ## to get g in the shape [batch, seq_len, attr_encoding_dim]
            g = g.unsqueeze(1).repeat(1, seq_len, 1)
        elif self.attribute_encoder_type == 'onehot':
            g = self.attr_encoder(y_onehot)
            g = g.repeat(1, seq_len, 1)
       
        ht = self.blocks(torch.cat((x_exog, g), dim=2))
                
        return self.mlp(ht)  # [batch_size,1,1]


    def predict(self, x: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.forward(x)


    def loss(self, x, y):
        if self.continuous_head == 'mse':
            return F.mse_loss(x, y)
        elif self.continuous_head == 'gaussian_nll':
            return F.gaussian_nll_loss(x[:, :, 0].unsqueeze(2), y,
                                       F.softplus(x[:, :, 1].unsqueeze(2)) **2)


class EnergyPlusResNet(ResNet):
    def __init__(self,
                 attribute_encoder_type: str = 'text', # 'onehot' or 'text'
                 resnet_hidden_dim: int = 512,
                 resnet_num_blocks: int = 2,
                 exog_input_dim: int = 103,
                 onehot_attr_input_dim: int = 336,
                 mlp_hidden_dim = 128,
                 qoi_dim = 1, # dimension of sim output
                 text_encoder_name = "distilbert-base-uncased",
                 text_freeze_encoder: bool = False,
                 text_finetune_only_specific_layers: bool = True,
                 continuous_head = 'mse',
                 ignore_attributes = False,
                 timestamp_embedding_size = 32
    ):
        super().__init__(
            attribute_encoder_type,
            resnet_hidden_dim,
            resnet_num_blocks,
            exog_input_dim,
            onehot_attr_input_dim,
            mlp_hidden_dim,
            qoi_dim,
            text_encoder_name,
            text_freeze_encoder,
            text_finetune_only_specific_layers,
            continuous_head,
            ignore_attributes)
        

        self.day_of_year_encoding = TimeSeriesSinusoidalPeriodicEmbedding(timestamp_embedding_size) 
        self.day_of_week_encoding = TimeSeriesSinusoidalPeriodicEmbedding(timestamp_embedding_size)
        self.hour_of_day_encoding = TimeSeriesSinusoidalPeriodicEmbedding(timestamp_embedding_size)


    def forward(self, batch: Dict):

        # if inputs are non-sequence, treat as vectors
        if len(batch["temperature"].shape) == 2:
            concat_dim = 1
        else: # treat as sequences
            concat_dim = 2

        # [batch_size, D]
        x_exog = torch.cat([
            self.day_of_year_encoding(batch['day_of_year']),
            self.day_of_week_encoding(batch['day_of_week']),
            self.hour_of_day_encoding(batch['hour_of_day']),
            batch["temperature"],
            batch["humidity"],
            batch["wind_speed"],
            batch["wind_direction"],
            batch["global_horizontal_radiation"],
            batch["direct_normal_radiation"],
            batch["diffuse_horizontal_radiation"]
        ], dim=concat_dim)

        y_onehot = batch["attributes_onehot"]
        y_text = {'input_ids': batch["attributes_input_ids"],
                    'attention_mask': batch["attributes_attention_mask"]}  

        return super().forward(x_exog, y_onehot, y_text)       


class WindResNet(ResNet):
    def __init__(self,
                 attribute_encoder_type: str = 'onehot', # 'onehot' or 'text'
                 resnet_hidden_dim: int = 512,
                 resnet_num_blocks: int = 2,
                 exog_input_dim: int = 103,
                 onehot_attr_input_dim: int = 336,
                 mlp_hidden_dim = 128,
                 qoi_dim = 1, # dimension of sim output
                 text_encoder_name = "distilbert-base-uncased",
                 text_freeze_encoder: bool = False,
                 text_finetune_only_specific_layers: bool = True,
                 continuous_head = 'mse',
                 ignore_attributes = False
    ):
        super().__init__(
            attribute_encoder_type,
            resnet_hidden_dim,
            resnet_num_blocks,
            exog_input_dim,
            onehot_attr_input_dim,
            mlp_hidden_dim,
            qoi_dim,
            text_encoder_name,
            text_freeze_encoder,
            text_finetune_only_specific_layers,
            continuous_head,
            ignore_attributes)


    def forward(self, batch: Dict):

        # [batch_size, D]
        x_exog = torch.cat([
            batch["wind_speed"],
            batch["wind_direction"],
            batch["turbulence_intensity"]
        ], dim=1)

        y_onehot = batch["attributes_onehot"]
        y_text = {'input_ids': batch["attributes_input_ids"],
                    'attention_mask': batch["attributes_attention_mask"]}  

        return super().forward(x_exog, y_onehot, y_text).reshape(-1,1)       
