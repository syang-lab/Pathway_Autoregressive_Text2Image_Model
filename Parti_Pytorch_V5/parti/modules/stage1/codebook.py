from typing import Tuple

import torch
import torch.nn as nn


class Codebook(nn.Module):
    """
        input
            encoded_image:[batch_size,seq_len,latent_dim]
        output
            z_q:encoded_image vectors in code book [batch_size,seq_len,latent_dim]
            min_encoding_indices: index in code book
            loss: code book loss
    """
    def __init__(self,codebook_vocab_size:int,latent_dim:int,beta:float):
        super(Codebook, self).__init__()
        self.codebook_vocab_size = codebook_vocab_size
        self.latent_dim = latent_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.codebook_vocab_size, self.latent_dim)
        self.embedding.weight.data.uniform_(-1.0/self.codebook_vocab_size, 1.0/self.codebook_vocab_size)


    def forward(self, z: torch.FloatTensor)->Tuple[torch.FloatTensor, torch.IntTensor, float]:
        z_flattened = z.view(-1, self.latent_dim)
        
        d = torch.sum(z_flattened**2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - \
            2*(torch.matmul(z_flattened, self.embedding.weight.t()))
        
        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)
        
        loss = torch.mean((z_q.detach() - z)**2) + self.beta * torch.mean((z_q - z.detach())**2)

        z_q = z + (z_q - z).detach()
        return z_q, min_encoding_indices, loss
