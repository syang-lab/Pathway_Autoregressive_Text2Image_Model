from typing import Optional, Tuple, Union, List, Any
from omegaconf import OmegaConf

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import lr_scheduler
import torch.nn.functional as F
import pytorch_lightning as pl

from utils.general import initialize_from_config

class PartiTransformer(pl.LightningModule):
    def __init__(self, sos_token:int, image_seq_len:int, text_key: str, stage1: OmegaConf, transformer: OmegaConf,
                path: Optional[str] = None, ignore_keys: List[str] = list(), scheduler: Optional[OmegaConf] = None) -> None:
    
        super( ).__init__()

        self.sos_token = sos_token
        self.image_seq_len = image_seq_len
        self.text_key = text_key
        
        # load stage1 model
        self.stage1_model = initialize_from_config(stage1)
        
        # make the parameters in stage1 model not trainable
        self.stage1_model.eval()
        for p in self.stage1_model.parameters():
            p.requires_grad = False

        # init transformer
        self.transformer = initialize_from_config(transformer)        
        
        if path is not None:
            self.init_from_ckpt(path, ignore_keys)
    

    def init_from_ckpt(self, path: str, ignore_keys: List[str] = list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")


    @torch.no_grad()
    def image_to_token(self, image: torch.FloatTensor)->Tuple[torch.FloatTensor, torch.IntTensor]:
        image_token = self.stage1_model.encode_codes(image)
        image_token = image_token.view(image.shape[0], -1)
        return image_token


    @torch.no_grad()
    def token_to_image(self, image_token: torch.IntTensor)->torch.FloatTensor:
        image = self.stage1_model.decode_codes(image_token)
        return image
    

    def forward(self, x_images: torch.FloatTensor, x_texts_token: torch.IntTensor)->Tuple[torch.FloatTensor, torch.IntTensor, torch.FloatTensor]:
        with torch.no_grad():
            x_images_token  = self.image_to_token(x_images)
        targets = x_images_token
        
        sos_tokens = torch.ones(x_images_token.shape[0], 1) * self.sos_token
        sos_tokens = sos_tokens.long().to(x_images.device)
        x_images_token = torch.cat((sos_tokens, x_images_token), dim=1)[:,:targets.size()[1]]
        

        logits = self.transformer(x_images_token, x_texts_token)
        
        return logits, targets
    

    def shared_step(self, batch: Tuple[Any, Any], batch_idx: int) -> torch.FloatTensor:
        x_images = batch[self.stage1_model.image_key]
        x_texts_token = batch[self.text_key]
        
        logits, targets = self(x_images, x_texts_token)
        loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.view(-1))
        
        return loss


    def training_step(self, batch: Tuple[Any, Any], batch_idx: int, optimizer_idx: int = 0) -> torch.FloatTensor:
        loss = self.shared_step(batch, batch_idx)
        self.log("train/total_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss


    def validation_step(self, batch: Tuple[Any, Any], batch_idx: int) -> torch.FloatTensor:
        loss = self.shared_step(batch, batch_idx)
        self.log("val/total_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss


    def configure_optimizers(self) -> torch.optim.Optimizer:
        """
        Following minGPT:
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.transformer.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)
                elif 'time_' in pn: # for RWKV
                    no_decay.add(fpn)

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.transformer.named_parameters()}
        inter_params = decay & no_decay 
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay/ignored set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": 0.01},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = [torch.optim.Adam(optim_groups, lr=self.learning_rate, betas=(0.9, 0.96))]
        scheduler = []

        if self.scheduler is not None:
            self.scheduler.params.start = self.learning_rate
            scheduler = initialize_from_config(self.scheduler)

            scheduler = [{
                'scheduler': lr_scheduler.LambdaLR(optimizer[0], lr_lambda=self.scheduler.schedule),
                'interval': 'step',
                'frequency': 1
            }]
        
        return optimizer, scheduler


    def top_k_logits(self, logits: torch.FloatTensor, k: int)->torch.FloatTensor:
        v, idx = torch.topk(logits, k)
        out = logits.clone()
        out[out < v[..., [-1]]] = -float("inf")
        return out


    @torch.no_grad()
    def sample(self, x_texts_token: torch.IntTensor, temperature: float=1.0, top_k: int=100, use_fp16: bool = True)->torch.IntTensor:
        sos_tokens = torch.ones(x_texts_token.shape[0], 1) * self.sos_token
        sos_tokens = sos_tokens.long().to(x_texts_token.device)
        x_images_token=sos_tokens
        
        self.transformer.eval()
        memory=self.transformer.encode(x_texts_token)

        for k in range(self.image_seq_len):
            logits = self.transformer.decode(memory, x_images_token)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                logits = self.top_k_logits(logits, top_k)
           
            probs = F.softmax(logits, dim=-1)

            images_token = torch.multinomial(probs, num_samples=1)
            x_images_token = torch.cat((x_images_token, images_token), dim=1)

        return x_images_token[:,1:]


    def log_images(self, x_images: torch.FloatTensor, x_texts_token: torch.IntTensor , temperature: float=1.0, top_k: int=100):
        log = dict()
        x_images_token = self.sample(x_texts_token, temperature, top_k)
        x_rec = self.token_to_image(x_images_token)

        log["input"] = x_images
        log["rec"] = x_rec

        return log, torch.cat((x_images, x_rec))