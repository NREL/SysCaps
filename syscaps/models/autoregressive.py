import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple, Dict, List
from syscaps.models.base_model import BaseSurrogateModel
from syscaps.models.modules import TimeSeriesSinusoidalPeriodicEmbedding
from syscaps.models.modules import RecurrentEncoder, TextEncoder, OneHotAttributeEncoder, IgnoreOneHotAttributes

def build(dataset, **kwargs):
    """Builds a EnergyPlusAutoregressive model."""
    return EnergyPlusAutoregressive(**kwargs)
    
        
class Autoregressive(BaseSurrogateModel):
    """ Autoregressive model for learning to map sim inputs to outputs.

        Supports both rnn and ssm autoregressive models.
        Supports both onehot and text attributes.

        Uses default PyTorch param initialization for Linear layers.
    """
    def __init__(self, 
                 attribute_encoder_type: str, # 'onehot' or 'text'                 
                 autoreg_type: str, # 'rnn' or 'ssm'
                 autoreg_input_dim: int,
                 autoreg_hidden_size: int,
                 autoreg_num_layers: int,
                 ssm_pool: List[int],
                 onehot_attr_input_dim: int,
                 mlp_hidden_dim: int,
                 qoi_dim: int, # dimension of sim output
                 text_encoder_name: str,
                 text_freeze_encoder: bool,
                 text_finetune_only_specific_layers: bool,
                 continuous_head: str,
                 ignore_attributes: bool,
                 is_autoregressive: bool = True):
        """
        Args:
            attribute_encoder_type (str): 'onehot' or 'text'
            autoreg_type (str): 'rnn' or 'ssm'
            autoreg_input_dim (int): dimension of the input to the autoreg model
            autoreg_hidden_size (int): dimension of the hidden state of the autoreg model
            autoreg_num_layers (int): number of layers in the autoreg model
            onehot_attr_input_dim (int): dimension of the onehot attribute input
            mlp_hidden_dim (int): dimension of the hidden layer in the MLP
            qoi_dim (int): dimension of the sim output
            text_encoder_name (str): name of the text encoder
            text_freeze_encoder (bool): whether to freeze the text encoder
            continuous_head (str): 'mse' or 'gaussian_nll'
        """
        super(Autoregressive,self).__init__(is_autoregressive)
        self.attribute_encoder_type = attribute_encoder_type
        
        if attribute_encoder_type == 'onehot':
            # use the same embedding dim as the text encoder would
            attr_output_dim = TextEncoder(
                model_name = text_encoder_name,
                freeze = text_freeze_encoder
            ).output_dim

            if not ignore_attributes:
                self.attr_encoder = OneHotAttributeEncoder(
                    input_dim= onehot_attr_input_dim,
                    output_dim = attr_output_dim
                )
            else:
                self.attr_encoder = IgnoreOneHotAttributes(
                    input_dim= onehot_attr_input_dim,
                    output_dim = attr_output_dim
                )

        elif attribute_encoder_type == 'text':
            self.attr_encoder = TextEncoder(
                model_name = text_encoder_name,
                freeze = text_freeze_encoder,
                finetune_only_specific_layers = text_finetune_only_specific_layers
            )
            attr_output_dim = self.attr_encoder.output_dim

        self.autoreg_encoder = RecurrentEncoder(
            autoreg_type = autoreg_type,
            pool = ssm_pool,
            input_dim = autoreg_input_dim + attr_output_dim,
            output_dim = autoreg_hidden_size,
            num_layers = autoreg_num_layers
        )

        # mlp + linear readout
        self.continuous_head = continuous_head
        out_dim = qoi_dim if self.continuous_head == 'mse' else qoi_dim*2        
        self.mlp = nn.Sequential(
            nn.Linear(autoreg_hidden_size, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, out_dim)
        )     
        
    def forward(self, 
                x_exog,
                y_onehot,
                y_text: Dict):
        """
        Args:
            x_exog (torch.Tensor) of shape [batch_size, sequence_length, feature_dim]
            y_onehot (torch.Tensor) of shape [batch_size, sequence_length, onehot_feature_dim]
            y_text (Dict) of with keys 'input_ids' and 'attention_mask', each 
                torch.Tensors   

        Returns:
            (torch.Tensor) of shape [batch_size, sequence_length, qoi_dim] if continuous_head == 'mse',
            (torch.Tensor) of shape [batch_size, sequence_length, 2*qoi_dim] if continuous_head == 'gaussian_nll'
        """
        seq_len = x_exog.shape[1]

        if self.attribute_encoder_type == 'text':
            g = self.attr_encoder(y_text['input_ids'], y_text['attention_mask'])
            # to get g in the shape [batch, seq_len, attr_encoding_dim]
            g = g.unsqueeze(1).repeat(1, seq_len, 1)
        elif self.attribute_encoder_type == 'onehot':
            g = self.attr_encoder(y_onehot)
            g = g.repeat(1, seq_len, 1)
       
        # either rnn or ssm
        ht = self.autoreg_encoder(torch.cat((x_exog, g), dim=2))
        return self.mlp(ht)

    def predict(self, x: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.forward(x)
    
    def loss(self, x, y):
        if self.continuous_head == 'mse':
            return F.mse_loss(x, y)
        elif self.continuous_head == 'gaussian_nll':
            return F.gaussian_nll_loss(x[:, :, 0].unsqueeze(2), y,
                                       F.softplus(x[:, :, 1].unsqueeze(2)) **2)
        
class EnergyPlusAutoregressive(Autoregressive):
    def __init__(self,
                 attribute_encoder_type: str = 'text', # 'onehot' or 'text'                 
                 autoreg_type: str = 'rnn', # 'rnn' or 'ssm'
                 autoreg_input_dim: int = 103,
                 autoreg_hidden_size = 256,
                 autoreg_num_layers = 1,
                 ssm_pool = [4,4],
                 onehot_attr_input_dim: int = 336,
                 mlp_hidden_dim = 128,
                 qoi_dim = 1, # dimension of sim output
                 text_encoder_name = "distilbert-base-uncased",
                 text_freeze_encoder: bool = False,
                 text_finetune_only_specific_layers: bool = True,
                 continuous_head: str = 'mse',
                 ignore_attributes = False,
                 timestamp_embedding_size = 32
    ):
        super().__init__(
            attribute_encoder_type,
            autoreg_type,
            autoreg_input_dim,
            autoreg_hidden_size,
            autoreg_num_layers,
            ssm_pool,
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
        # [batch_size, seq_len, D]
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
        ], dim=2)

        seq_len = x_exog.shape[1]
        if self.autoreg_encoder.autoreg_type == 'ssm':
            # pad to next multiple of N=16
            N=16
            pad_len = seq_len % N
            if (N-pad_len) > 0:
                # pad right
                x_exog = F.pad(x_exog, (0,0,0,N-pad_len), "constant", 0)

        y_onehot = batch["attributes_onehot"]
        y_text = None
        if 'attributes_input_ids' in batch:
            y_text = {'input_ids': batch["attributes_input_ids"],
                        'attention_mask': batch["attributes_attention_mask"]}  

        out = super().forward(x_exog, y_onehot, y_text)
        if self.autoreg_encoder.autoreg_type == 'ssm':
            out = out[:,:seq_len]
        return out