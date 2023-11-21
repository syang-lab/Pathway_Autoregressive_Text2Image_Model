import os
import numpy as np
from tqdm import tqdm
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
import torchvision.utils as vutils

from parti.utils.tokenizer import SimpleTokenizer
from parti.dataloader.textimage import TextImageTrain, TextImageValidation

from parti.modules.stage2.partitransformer import PartiTransformer

class Stage2(nn.Module):
    def __init__(self,
                 text_vocab_size,
                 dim_model, 
                 nhead, 
                 num_encoder_layers, 
                 num_decoder_layers, 
                 dim_feedforward, 
                 dropout, 
                 image_size, 
                 image_channels, 
                 patch_size, 
                 vit_dim_model, 
                 vit_nhead, 
                 vit_nlayer, 
                 vit_dim_feedforward, 
                 codebook_vocab_size, 
                 latent_dim, 
                 beta,
                 vqgan_checkpoint_file,
                 device):
        super().__init__()
        self.device=device
        self.image_size=image_size

        self.parti = PartiTransformer(text_vocab_size, dim_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout, 
                                      image_size, image_channels, patch_size, vit_dim_model, vit_nhead, vit_nlayer, vit_dim_feedforward, codebook_vocab_size, latent_dim, beta, 
                                      vqgan_checkpoint_file).to(self.device)
    
    
    def load_parti(self,parti_checkpoint_file):
        self.parti.load_state_dict(torch.load(parti_checkpoint_file))
        return 


    def prepare_training(self,output_path:str)->None:
        self.output_path=output_path
        os.makedirs(os.path.join(self.output_path,"results"), exist_ok=True)
        os.makedirs(os.path.join(self.output_path,"checkpoints"), exist_ok=True)
        return 


    def load_dataset(self, dataset_path:str, batch_size=2, text_seq_len=128):
        tokenizer=SimpleTokenizer(text_seq_len,truncate_captions=True)
        
        train_data=TextImageTrain(dataset_path, tokenizer,self.image_size)
        self.train_loader = DataLoader(train_data, batch_size, shuffle=False)

        valid_data=TextImageValidation(dataset_path, tokenizer,self.image_size)
        self.valid_loader = DataLoader(valid_data, batch_size, shuffle=False)
        return
            

    def optimizers(self,lr):
        self.optimizer = torch.optim.AdamW(self.parti.parameters(), lr=lr, betas=(0.9, 0.95),weight_decay=0.045)
        return 

    def train(self,epoch):
        self.parti.train()
        with tqdm(range(len(self.train_loader))) as pbar:
            for step, images_text in zip(pbar,self.train_loader):
                images=images_text[0]
                images = images.to(self.device)
                
                texts_token=images_text[1]
                texts_token = texts_token.to(self.device)

                logits, targets, _= self.parti(images, texts_token)

                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

                self.optimizer.zero_grad()
                loss.backward()

                self.optimizer.step()

                pbar.set_postfix(Transformer_Train_Loss=np.round(loss.cpu().detach().numpy().item(), 4))
                pbar.update(0)
                
                _, sampled_imgs = self.parti.log_images(images,texts_token,temperature=1.0, top_k=100)
                vutils.save_image(sampled_imgs, os.path.join(self.output_path+"/results", f"transformer_{epoch}{step}.jpg"), nrow=4)
        
        torch.save(self.parti.state_dict(), os.path.join(self.output_path+"/checkpoints", f"transformer_{epoch}.pt"))
        

    def valid(self,epoch):
        self.parti.eval()
        with tqdm(range(len(self.valid_loader))) as pbar:
            for step, images_text in zip(pbar,self.train_loader):
                images = images_text[0]
                images = images.to(self.device)
                
                texts_token = images_text[1]
                texts_token = texts_token.to(self.device)

                logits, targets, _= self.parti(images, texts_token)

                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

                pbar.set_postfix(Transformer_Valid_Loss=np.round(loss.cpu().detach().numpy().item(), 4))
                pbar.update(0)
        

    def train_valid(self,epochs):
        for epoch in range(epochs):
            self.train(epoch)
            self.valid(epoch)
        return


    def inference(self, texts_token, temperature=1.0, top_k=100):
        images_token=self.parti.sample(texts_token, temperature, top_k)
        images=self.parti.token_to_image(images_token)
        
        vutils.save_image(images, os.path.join(self.output_path,"inference", "transformer.jpg"), nrow=4)
        return images
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="stage2: parti")

    args = parser.parse_args()
    #set up device
    args.device="cpu"
    args.output_path="/assets"
    args.vqgan_checkpoint_file="/assets/checkpoints/vqgan_epoch_0.pt"

    #VitEncoder and VitDecoder
    args.image_size=(64,64)
    args.image_channels=3
    args.patch_size=(8,8)
    args.vit_dim_model=32
    args.vit_nhead=2
    args.vit_nlayer=2
    args.vit_dim_feedforward=4*args.vit_dim_model
    
    #Codebook
    args.codebook_vocab_size=128
    args.latent_dim=args.vit_dim_model
    args.beta=0.25

    
    #Transformer
    args.text_vocab_size=49408
    args.dim_model=64
    args.nhead=8
    args.num_encoder_layers=1
    args.num_decoder_layers=1
    args.dim_feedforward=1
    args.dropout=0.1
    

    #Build Model
    stage2=Stage2(args.text_vocab_size,
                 args.dim_model, 
                 args.nhead, 
                 args.num_encoder_layers, 
                 args.num_decoder_layers, 
                 args.dim_feedforward, 
                 args.dropout, 
                 args.image_size, 
                 args.image_channels, 
                 args.patch_size, 
                 args.vit_dim_model, 
                 args.vit_nhead, 
                 args.vit_nlayer, 
                 args.vit_dim_feedforward, 
                 args.codebook_vocab_size, 
                 args.latent_dim, 
                 args.beta,
                 args.vqgan_checkpoint_file,
                 args.device)
    

    #Make directory
    stage2.prepare_training(args.output_path)
    

    #Load Dataset
    args.dataset_path=r"/assets/flowers"
    args.batch_size=2
    args.text_seq_len=128
    stage2.load_dataset(args.dataset_path, args.batch_size, args.text_seq_len)
    

    #Optimizer
    args.learning_rate=0.0001
    stage2.optimizers(args.learning_rate)


    #Train and Valid
    args.epochs=3
    stage2.train_valid(args.epochs)
