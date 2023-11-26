import math
import torch
import torch.nn as nn
from torch.nn import Transformer  as BaseTransformer


class PositionalEncoding(nn.Module):
    def __init__(self, dim_model:int, max_len: int=1024)->None:
        """
            parameters
                dim_model: embedding dimension
                max_len: max sequence length
            input
                embeded_images:[batch_size, seq_len, dim_model]
            output
                pe: positional embedding [1,max_len, dim_model]
        """
        super().__init__()

        # Create matrix of [seqLen, dim_model] representing the positional encoding for max_len inputs
        pe = torch.zeros(max_len, dim_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim_model, 2).float() * (-math.log(10000.0)/dim_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        # register_buffer => Tensor which is not a parameter, but should be part of the modules state.
        # Used for tensors that need to be on the same device as the module.
        # persistent=False tells PyTorch to not add the buffer to the state dict (e.g. when we save the model)
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x: torch.FloatTensor)->torch.FloatTensor:
        return self.pe[:, :x.size(1),:]
    

class Transformer(nn.Module):
    def __init__(self, text_vocab_size: int , image_vocab_size: int, dim_model: int, nhead: int, num_encoder_layers: int, num_decoder_layers: int, dim_feedforward: int, dropout: float):
        super().__init__()
        self.positional_encoding=PositionalEncoding(dim_model)

        self.encoder_embedding=nn.Embedding(text_vocab_size, dim_model)
        self.encoder_embedding.weight.data.normal_(mean=0.0, std=0.02)
        self.decoder_embedding=nn.Embedding(image_vocab_size, dim_model)
        self.encoder_embedding.weight.data.normal_(mean=0.0, std=0.02)

        self.transformer=BaseTransformer(dim_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout)

        self.linear=nn.Linear(dim_model, image_vocab_size)
        self.linear.weight.data.normal_(mean=0.0, std=0.02)


    def _make_tgt_mask(self,tgt: torch.IntTensor)->torch.IntTensor:
        """
        Args:
            trg: target sequence
        Returns:
            trg_mask: target mask
        """
        batch_size, tgt_len = tgt.size()
        # returns the lower triangular part of matrix filled with ones
        tgt_mask = torch.tril(torch.ones((tgt_len, tgt_len)))
        return tgt_mask
        

    def forward(self,x_images_token: torch.IntTensor, x_texts_token: torch.IntTensor)->torch.FloatTensor:
        x_texts_emb=self.encoder_embedding(x_texts_token)
        x_texts_emb=x_texts_emb+self.positional_encoding(x_texts_token)

        x_images_mask=self._make_tgt_mask(x_images_token)
        x_images_emb=self.decoder_embedding(x_images_token)
        x_images_emb=x_images_emb+self.positional_encoding(x_images_token)
        

        output=self.transformer(src=torch.transpose(x_texts_emb,0,1), tgt=torch.transpose(x_images_emb,0,1), tgt_mask=x_images_mask)
        output=torch.transpose(output,0,1)
        output=self.linear(output)
        return output


    def encode(self, x_texts_token: torch.IntTensor)->torch.FloatTensor:
        x_texts_emb=self.encoder_embedding(x_texts_token)
        x_texts_emb=x_texts_emb+self.positional_encoding(x_texts_token)

        x_texts_encode=self.transformer.encoder(torch.transpose(x_texts_emb,0,1))

        x_texts_encode=torch.transpose(x_texts_encode,0,1)
        return x_texts_encode


    def decode(self, x_texts_encode: torch.FloatTensor, x_images_token: torch.IntTensor)->torch.FloatTensor:
        x_images_emb=self.decoder_embedding(x_images_token)
        x_images_emb=x_images_emb+self.positional_encoding(x_images_token)

        output=self.transformer.decoder(tgt=torch.transpose(x_images_emb,0,1), memory=torch.transpose(x_texts_encode,0,1))

        output=torch.transpose(output,0,1)
        output=self.linear(output)
        return output