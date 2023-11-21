from typing import Union, Tuple

import torch
import torch.nn as nn

from parti.modules.stage1.vit import ViTEncoder, ViTDecoder
from parti.modules.stage1.codebook import Codebook


class VQGAN(nn.Module):
    def __init__(self, 
                 image_size: Union[Tuple[int, int], int],
                 image_channels: int, 
                 patch_size: Union[Tuple[int, int], int], 
                 dim_model: int, 
                 nhead: int, 
                 nlayer: int, 
                 dim_feedforward: int, 
                 codebook_vocab_size: int,
                 latent_dim: int, 
                 beta: float)->None:
        
        super(VQGAN, self).__init__()
        self.encoder = ViTEncoder(image_size,image_channels, patch_size, dim_model, nhead , nlayer, dim_feedforward)
        self.decoder = ViTDecoder(image_size,image_channels, patch_size, dim_model, nhead , nlayer, dim_feedforward)
        self.codebook = Codebook(codebook_vocab_size,latent_dim,beta)

        self.pre_quant = nn.Linear(dim_model, latent_dim)
        self.post_quant = nn.Linear(dim_model, latent_dim)


    def forward(self, x: torch.FloatTensor)->Tuple[torch.FloatTensor, torch.FloatTensor, float]:        
        z_q, z_indices, q_loss=self.encode(x)
        decoded_z_q=self.decode(z_q)
        return decoded_z_q, z_indices, q_loss
    

    def encode(self, x: torch.FloatTensor)->Tuple[torch.FloatTensor, torch.FloatTensor, float]:
        encoded_x = self.encoder(x)
        pre_quant_x = self.pre_quant(encoded_x)
        z_q, z_indices, q_loss = self.codebook(pre_quant_x)
        return z_q, z_indices, q_loss


    def decode(self, z_q: torch.FloatTensor)->torch.FloatTensor:
        post_quant_z_q = self.post_quant(z_q)
        decoded_z_q = self.decoder(post_quant_z_q)
        return decoded_z_q


    def calculate_lambda(self, perceptual_loss: float, gan_loss: float)->float:
        last_layer = self.decoder.to_pixel[-1]
        last_layer_weight = last_layer.weight
        perceptual_loss_grads = torch.autograd.grad(perceptual_loss, last_layer_weight, retain_graph=True)[0]
        gan_loss_grads = torch.autograd.grad(gan_loss, last_layer_weight, retain_graph=True)[0]

        λ = torch.norm(perceptual_loss_grads) / (torch.norm(gan_loss_grads) + 1e-4)
        λ = torch.clamp(λ, 0, 1e4).detach()
        return 0.8 * λ


    @staticmethod
    def adopt_weight(disc_factor: float, step: int, threshold: int, value: float=0.1)->float:
        if step < threshold:
            disc_factor = value
        return disc_factor
    







