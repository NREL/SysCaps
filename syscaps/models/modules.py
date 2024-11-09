import torch
from torch import nn
import math
import numpy as np
from typing import List
from syscaps.models.third_party.sashimi import Sashimi # SSM
from transformers import DistilBertModel,  BertModel, LongformerForSequenceClassification, LongformerConfig
#DistilBertTokenizer, BertTokenizer, 


class TimeSeriesSinusoidalPeriodicEmbedding(nn.Module):
    """This module produces a sinusoidal periodic embedding for a sequence of values in [-1, +1]."""
    def __init__(self, embedding_dim: int) -> None:
        """
        Args:
            embedding_dim (int): embedding size.
        """
        super().__init__()
        self.linear = nn.Linear(2, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """`x` is expected to be [batch_size, seqlen, 1]."""
        with torch.no_grad():
            x = torch.cat([torch.sin(np.pi * x), torch.cos(np.pi * x)], dim=2)
        # [batch_size, seqlen x 2] --> [batch_size, seqlen, embedding_dim]
        return self.linear(x)


class SequenceEncoder(nn.Module):
    """
    Recurrent encoder for time series inputs.
    Preserves the length of time series.

    Example: X may be text features, timestamp, and weather time series of shape [batch_size, sequence_length, input_dim],
    where input_dim = text feature dim + # timestamp features + # weather features.

    Supports both LSTM and SSM (Sashimi) bi-directional recurrent encoders.
    """
    def __init__(
        self,
        input_dim: int,  # conventions: use 'dim' over 'size'
        output_dim: int = 128, # use 'input'/'output' if appropriate rather than 'hidden'
        num_layers: int = 1, 
        pool: List[int] = [4,4],
        seq_type = 'rnn'
    ):
        """
        Args:
            seq_type (str): 'rnn' or 'ssm'
            input_dim (int): size of the input feature dim
            output_dim (int): size of the hidden state of the recurrent encoder
            num_layers (int): number of recurrent layers
        """
        super().__init__()
        self.num_layers  = num_layers
        self.seq_type = seq_type
        self.output_dim = output_dim

        # bidirectional LSTM 
        if seq_type == 'rnn':
            self.encoder = nn.LSTM(input_dim,
                                output_dim, num_layers=self.num_layers,
                                batch_first=True, bidirectional=True)
        # bidirectional SSM
        elif seq_type == 'ssm':
            self.ssm_in = nn.Linear(input_dim, output_dim)
            self.encoder = Sashimi(
                d_model=output_dim,
                n_layers=self.num_layers,
                pool=pool, # no down/up sampling layers
                bidirectional=True
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): input of shape [batch_size, seq_len, input_dim]

        Returns:
            x_embedding (torch.Tensor): embedding of shape [batch_size, seq_len, output_dim]
        """

        # need to project the input to the dim of the SSM encoder.
        # the SSM does not change the size of the last dim of the input.
        if self.seq_type == 'ssm':
            x = self.ssm_in(x) # [batch_size, seq_len, output_dim]

        # x is shape [batch_size, seq_len, dim]
        outs, _ = self.encoder(x)
        
        if self.seq_type == 'rnn':
            # outs is shape [batch_size, seq_len, 2*output_dim]            
            # take the average of hidden states for both directions to get an embedding for x
            x_embedding = torch.cat([
                outs[:,:,:self.output_dim].unsqueeze(3),
                outs[:,:,self.output_dim:].unsqueeze(3)
            ], dim=3).mean(dim=3)
        else:
            # outs is shape [batch_size, seq_len, output_dim]            
            # sashimi handles bidirectional fusion internally
            x_embedding = outs

        return x_embedding 
    

class TextEncoder(nn.Module):
    """
    Encode attribute caption Y_hat to a fixed size vector.
    Uses CLS token hidden representation as the sentence's embedding.
    """
    def __init__(
        self,
        model_name: str = "distilbert-base-uncased", 
        freeze: bool = True,
        finetune_only_specific_layers: bool = True,
    ):
        """
        Args:
            model_name (str): name of the pre-trained model to use
            freeze (bool): True -> freeze all text encoder parameters,
                           False -> allow all text encoder parameters to be trained
            finetune_only_specific_layers (bool): True -> only train last layer, 
                             False -> has no effect
        """
        super().__init__()
        self.model_name = model_name
        self.freeze = freeze
        self.finetune_only_specific_layers = finetune_only_specific_layers
        # we are using the CLS token hidden representation as the sentence's embedding
        # n.b. every tokenized caption has [CLS] token at the first position
        # Example: [CLS] Sentence A [SEP] Sentence B [SEP]
        self.target_token_idx = 0
        if self.freeze:
            print('[WARNING] ALL parameters of the text encoder are frozen!!!')

        if model_name == "distilbert-base-uncased":
                        
            self.model = DistilBertModel.from_pretrained(model_name)
            #self.tokenizer = DistilBertTokenizer.from_pretrained(model_name, max_length=model_max_length)
            
            if self.freeze:
                for name, param in self.model.named_parameters():
                    param.requires_grad = False
            else:
                for name, param in self.model.named_parameters():
                    # if finetuning, freeze all layers not in transformer.layer.5
                    if "transformer.layer.5" not in name and self.finetune_only_specific_layers: 
                        param.requires_grad = False
                    # not freezing, and not finetuning - so requires_grad!
                    elif "transformer.layer.5" not in name:
                        param.requires_grad = True
                    # not freezing and finetuning - tune the last layer
                    elif self.finetune_only_specific_layers:
                        param.requires_grad = True
                    # not freezing, and not finetuning - tune the last layer too
                    else:
                        param.requires_grad = True
                    
        elif model_name == "bert-base-uncased":
            self.model = BertModel.from_pretrained(model_name,
                                                   output_hidden_states=True)
            #self.tokenizer = BertTokenizer.from_pretrained(model_name, max_length=model_max_length)
            self.projection = nn.Linear(4 * self.model.config.hidden_size,
                                        self.model.config.hidden_size)

            if self.freeze:
                for name, param in self.model.named_parameters():
                    param.requires_grad = False
            else:
                layers_to_finetune = [f'encoder.layer.{i}' for i in range(8,12)]
                for name, param in self.model.named_parameters():
                    # if finetuning, freeze all layers except the last four
                    if name not in layers_to_finetune and self.finetune_only_specific_layers: 
                        param.requires_grad = False
                    # not freezing, and not finetuning - so requires_grad!
                    elif name not in layers_to_finetune:
                        param.requires_grad = True
                    # not freezing and finetuning - tune the last layers
                    elif self.finetune_only_specific_layers:
                        param.requires_grad = True
                    # not freezing, and not finetuning - tune the last layer too
                    elif name:
                        param.requires_grad = True
                # always disable pooler params because we don't use it
                for name, param in self.model.pooler.named_parameters():
                    param.requires_grad = False

        elif model_name == "longformer-base-4096":
            # We use SequenceClassification variant which implements
            # global attention for the CLS token.
            # We use the default sliding attention window size of 512.
            self.model = LongformerForSequenceClassification.from_pretrained('allenai/longformer-base-4096', 
                                                         output_hidden_states = True,
                                                         gradient_checkpointing = True)
            config = LongformerConfig.from_pretrained('allenai/longformer-base-4096')
            config.max_position_embeddings = 4096
            self.model.config = config

            self.projection = nn.Linear(4 * self.model.config.hidden_size,
                                        self.model.config.hidden_size)
            if self.freeze:
                for name, param in self.model.named_parameters():
                    param.requires_grad = False
            else:
                layers_to_finetune = [f'encoder.layer.{i}' for i in range(8,12)]
                for name, param in self.model.named_parameters():
                    # if finetuning, freeze all layers except the last four
                    if name not in layers_to_finetune and self.finetune_only_specific_layers: 
                        param.requires_grad = False
                    # not freezing, and not finetuning - so requires_grad!
                    elif name not in layers_to_finetune:
                        param.requires_grad = True
                    # not freezing and finetuning - tune the last layers
                    elif self.finetune_only_specific_layers:
                        param.requires_grad = True
                    # not freezing, and not finetuning - tune the last layer too
                    elif name:
                        param.requires_grad = True
                # always disable classifier params because we don't use it
                for name, param in self.model.classifier.named_parameters():
                    param.requires_grad = False

        self.output_dim = self.model.config.hidden_size


    def forward(self, input_ids, attention_masks) -> torch.Tensor:
        """
        Args:
            input_ids (torch.LongTensor): input of shape [batch_size, seq_len]
            attention_masks (torch.Tensor): attention masks of shape [batch_size, seq_len]

        Returns:
            last_hidden_state (torch.Tensor): embedding of shape [batch_size, output_dim]
        """
        if self.model_name == 'distilbert-base-uncased':
            output = self.model(input_ids=input_ids,
                                attention_mask=attention_masks)
            last_hidden_state = output.last_hidden_state
            return last_hidden_state[:, self.target_token_idx, :]
        elif self.model_name == 'bert-base-uncased' or 'longformer-base-4096':
            outputs = self.model(input_ids=input_ids,
                                attention_mask=attention_masks)
            
            last_hidden_states = outputs.hidden_states
            last_hidden_states = torch.cat(tuple([last_hidden_states[i] for i in [-4, -3, -2, -1]]), dim=-1)
            return self.projection(last_hidden_states[:, self.target_token_idx, :])
 

class OneHotAttributeEncoder(nn.Module):
    """
    Embed attributes represented as concatenated one-hot vectors.
    """
    def __init__(self, input_dim: int = 1, output_dim: int = 128):
        """
        Args:
            input_dim (int): dimension of the input vector
            output_dim (int): dimension of the output vector
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): one-hot attributes as single vector of shape [batch_size, input_dim]
        
        Returns:
            (torch.Tensor) of shape [batch_size, output_dim]
        """
        return self.mlp(x)
    

class IgnoreOneHotAttributes(nn.Module):
    """
    Returns tensor of zeros instead of the embedded one-hot vector
    """
    def __init__(self, input_dim: int = 1, output_dim: int = 128):
        """
        Args:
            input_dim (int): dimension of the input vector
            output_dim (int): dimension of the output vector
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.zeros = torch.zeros(1, 1, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): one-hot attributes as single vector of shape [batch_size, seq_len, input_dim]
        
        Returns:
            (torch.Tensor) of shape [batch_size, seq_len, output_dim]
        """
        return self.zeros.repeat(x.shape[0], x.shape[1], 1).to(x.device)
    

class ResBlock(nn.Module):
    """Basic residual block.
    """
    def __init__(self, in_dim, out_dim, downsample=None):
        super(ResBlock, self).__init__()
        self.activation = nn.GELU()
        self.linear1 = nn.Linear(in_dim, out_dim)
        self.linear2 = nn.Linear(out_dim, out_dim)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        
        out = self.linear1(x)
        out = self.activation(out)
        out = self.linear2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.activation(out)

        return out
    

class PositionalEncoding(nn.Module):
    """Helper Module that adds positional encoding to the token embedding to
    introduce a notion of order within a time-series.
    """
    def __init__(self,
                emb_size: int,
                dropout: float,
                maxlen: int = 500):
        """
        Args:
            emb_size (int): embedding size.
            dropout (float): dropout rate.
            maxlen (int): maximum possible length of the incoming time series.
        """
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: torch.Tensor) -> torch.Tensor:
        # batch first - use size(1)
        # need to permute token embeddings from [batch_size, seqlen x emb_size] to [seqlen x batch_size, emb_size]
        return self.dropout(token_embedding.permute(1,0,2) + self.pos_embedding[:token_embedding.size(1), :]).permute(1,0,2)
