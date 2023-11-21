from typing import Union, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from parti.modules.stage1.vqgan import VQGAN
from parti.modules.stage2.transformer import Transformer
from parti.modules.stage1.vqgan import VQGAN


class PartiTransformer(nn.Module):
    def __init__(self, 
                 text_vocab_size: int,
                 dim_model: int, 
                 nhead: int, 
                 num_encoder_layers: int, 
                 num_decoder_layers: int, 
                 dim_feedforward: int, 
                 dropout: int, 
                 image_size: Union[Tuple[int, int], int],
                 image_channels: int, 
                 patch_size: Union[Tuple[int, int], int], 
                 vit_dim_model: int, 
                 vit_nhead: int, 
                 vit_nlayer: int, 
                 vit_dim_feedforward: int, 
                 codebook_vocab_size: int, 
                 latent_dim: int, 
                 beta: float,
                 checkpoint_file: str)->None:
        
        super( ).__init__()

        self.sos_token=0
        self.image_seq_len=patch_size[0]*patch_size[1]

        self.vqgan = VQGAN(image_size, image_channels, patch_size, vit_dim_model, vit_nhead, vit_nlayer, vit_dim_feedforward, codebook_vocab_size, latent_dim, beta)
        self.load_vqgan(checkpoint_file)

        self.transformer = Transformer(text_vocab_size,codebook_vocab_size, dim_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout)
        self.linear=nn.Linear(dim_model, codebook_vocab_size)


    def load_vqgan(self,checkpoint_file: str):
        self.vqgan.load_state_dict(torch.load(checkpoint_file))
        self.vqgan = self.vqgan.eval()
        return 


    @torch.no_grad()
    def image_to_token(self, image: torch.FloatTensor)->Tuple[torch.FloatTensor, torch.IntTensor]:
        quant_z, image_token, _ = self.vqgan.encode(image)
        image_token = image_token.view(quant_z.shape[0], -1)
        return quant_z, image_token


    @torch.no_grad()
    def token_to_image(self, image_token: torch.IntTensor)->torch.FloatTensor:
        image_token_embedding = self.vqgan.codebook.embedding(image_token)
        image = self.vqgan.decode(image_token_embedding)
        return image
    

    def forward(self, x_images: torch.FloatTensor, x_texts_token: torch.IntTensor)->Tuple[torch.FloatTensor, torch.IntTensor, torch.FloatTensor]:
        quant_z, x_images_token  = self.image_to_token(x_images)
        target = x_images_token
        

        sos_tokens = torch.ones(x_images_token.shape[0], 1) * self.sos_token
        sos_tokens = sos_tokens.long().to(x_images.device)
        x_images_token = torch.cat((sos_tokens, x_images_token), dim=1)[:,:target.size()[1]]
        

        logits = self.transformer(x_images_token, x_texts_token)
        logits = self.linear(logits)
        
        return logits, target, quant_z


    def top_k_logits(self, logits: torch.FloatTensor, k: int)->torch.FloatTensor:
        v, idx = torch.topk(logits, k)
        out = logits.clone()
        out[out < v[..., [-1]]] = -float("inf")
        return out


    @torch.no_grad()
    def sample(self, x_texts_token: torch.IntTensor, temperature: float=1.0, top_k: int=100)->torch.IntTensor:
        self.transformer.eval()
        self.linear.eval()
        self.vqgan.eval()

        sos_tokens = torch.ones(x_texts_token.shape[0], 1) * self.sos_token
        sos_tokens = sos_tokens.long().to(x_texts_token.device)
        x_images_token=sos_tokens
        
        memory=self.transformer.encode(x_texts_token)

        for k in range(self.image_seq_len):
            logits = self.transformer.decode(memory, x_images_token)
            logits = self.linear(logits)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                logits = self.top_k_logits(logits, top_k)
           
            probs = F.softmax(logits, dim=-1)

            images_token = torch.multinomial(probs, num_samples=1)
            x_images_token = torch.cat((x_images_token, images_token), dim=1)

        return x_images_token[:,1:]


    @torch.no_grad()
    def log_images(self, x_images: torch.FloatTensor, x_texts_token: torch.IntTensor , temperature: float=1.0, top_k: int=100):
        log = dict()
        x_images_token = self.sample(x_texts_token, temperature, top_k)
        x_rec = self.token_to_image(x_images_token)

        log["input"] = x_images
        log["rec"] = x_rec

        return log, torch.cat((x_images, x_rec))