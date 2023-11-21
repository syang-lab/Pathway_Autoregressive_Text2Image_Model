# ------------------------------------------------------------------------------------
# Modified from ViT-Pytorch (https://github.com/lucidrains/vit-pytorch)
# Copyright (c) 2020 Phil Wang. All Rights Reserved.
# ------------------------------------------------------------------------------------

import math
from typing import Union, Tuple

import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange


def init_weights(m:nn.Module)->None:
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        w = m.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))


class PositionalEncoding(nn.Module):
    def __init__(self, dim_model:int, max_len: int=5000)->None:
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


class PreNorm(nn.Module):
    def __init__(self, dim_model: int, fn: nn.Module) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim_model)
        self.fn = fn

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        x=self.norm(x)
        x=self.fn(x)
        return x


class FeedForward(nn.Module):
    def __init__(self, dim_model: int, dim_feedforward: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_model, dim_feedforward),
            nn.Tanh(),
            nn.Linear(dim_feedforward, dim_model)
        )

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim_model: int, nhead: int) -> None:
        super().__init__()
        self.dim_model=dim_model
        self.heads = nhead
        self.dim_head=dim_model//nhead
        self.dim_inner=self.dim_head*nhead
       
        self.scale = self.dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(self.dim_model, self.dim_inner * 3, bias = False)

        self.to_out = nn.Linear(self.dim_inner, self.dim_model)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        qkv = self.to_qkv(x).chunk(3, dim = -1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim_model: int, nhead : int, nlayer: int, dim_feedforward: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(nlayer):
            layer = nn.ModuleList([PreNorm(dim_model, Attention(dim_model, nhead)),
                                   PreNorm(dim_model, FeedForward(dim_model, dim_feedforward))])
            self.layers.append(layer)

        self.norm = nn.LayerNorm(dim_model)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)


class ViTEncoder(nn.Module):
    """
        parameters
            image_size
            image_channels
            patch_size
            dim_model
            nhead
            nlayer
            dim_feedforward
        input
            image:batch_size,image_channels,image_height,image_width
        output
            encoded_image:[batch_size,seq_len=patch_size[0]*patch_size[1],dim_model]
            
    """
    def __init__(self, image_size: Union[Tuple[int, int], int],image_channels: int, patch_size: Union[Tuple[int, int], int], dim_model: int, nhead: int, nlayer: int, dim_feedforward: int)->None:

        super().__init__()
        image_height, image_width = image_size if isinstance(image_size, tuple) \
                            else (image_size, image_size)
        patch_height, patch_width = patch_size if isinstance(patch_size, tuple) \
                                    else (patch_size, patch_size)

        self.to_patch_embedding = nn.Sequential(
            nn.Conv2d(image_channels, dim_model, kernel_size=patch_size, stride=patch_size),
            Rearrange('b c h w -> b (h w) c'),
        )

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        self.seq_len=(image_height//patch_height)*(image_width//patch_width)
        self.en_pos_embedding = PositionalEncoding(dim_model, self.seq_len)
    
        self.transformer = Transformer(dim_model, nhead, nlayer, dim_feedforward)
        self.apply(init_weights)


    def forward(self, x_img: torch.FloatTensor)->torch.FloatTensor:
        x = self.to_patch_embedding(x_img)
        x = x+self.en_pos_embedding(x)
        x = self.transformer(x)
        return x


class ViTDecoder(nn.Module):
    """
        parameters
            image_size
            image_channels
            patch_size
            dim_model
            nhead
            nlayer
            dim_feedforward
        input
            encoded_image:[batch_size,seq_len=patch_size[0]*patch_size[1],dim_model]
        
        output
            image:batch_size,image_channels,image_height,image_width            
    """

    def __init__(self, image_size: Union[Tuple[int, int], int],image_channels: int, patch_size: Union[Tuple[int, int], int], dim_model: int, nhead: int, nlayer: int, dim_feedforward: int)->None:
        super().__init__()
        image_height, image_width = image_size if isinstance(image_size, tuple) \
                            else (image_size, image_size)
        patch_height, patch_width = patch_size if isinstance(patch_size, tuple) \
                                    else (patch_size, patch_size)


        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        self.seq_len=(image_height//patch_height)*(image_width//patch_width)
        self.de_pos_embedding = PositionalEncoding(dim_model, self.seq_len)

        self.transformer = Transformer(dim_model, nhead, nlayer, dim_feedforward)
        
        self.to_pixel = nn.Sequential(
            Rearrange('b (h w) c -> b c h w', h=image_height // patch_height),
            nn.ConvTranspose2d(dim_model, image_channels, kernel_size=patch_size, stride=patch_size)
        )
        self.apply(init_weights)


    def forward(self, x_token: torch.FloatTensor) -> torch.FloatTensor:
        x = x_token - self.de_pos_embedding(x_token)
        x = self.transformer(x)
        x = self.to_pixel(x)
        return x


    def get_last_layer(self) -> nn.Parameter:
        return self.to_pixel[-1].weight    
