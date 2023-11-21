import os
import argparse
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
import torchvision.utils as vutils

from parti.utils.tokenizer import SimpleTokenizer
from parti.dataloader.textimage import TextImageTrain, TextImageValidation

from parti.losses.discriminator import Discriminator
from parti.losses.lpips import LPIPS

from parti.modules.stage1.vqgan import VQGAN


class Stage1(nn.Module):
    def __init__(self, image_size, image_channels, patch_size, dim_model, nhead, nlayer, dim_feedforward, codebook_vocab_size, latent_dim, beta, device):
        super().__init__()
        self.device=device
        self.image_size=image_size
        self.image_channels=image_channels
        
        
        self.vqgan=VQGAN(image_size, image_channels, patch_size, dim_model, nhead, nlayer, dim_feedforward, codebook_vocab_size, latent_dim, beta).to(self.device)
        self.discriminator = Discriminator(image_channels).to(self.device)
        self.perceptual_loss = LPIPS().eval().to(self.device)

   
    def load_dataset(self, dataset_path:str, batch_size=2, text_seq_len=128):
        tokenizer=SimpleTokenizer(text_seq_len,truncate_captions=True)
        train_data=TextImageTrain(dataset_path, tokenizer,self.image_size)
        self.train_loader = DataLoader(train_data, batch_size, shuffle=False)

        valid_data=TextImageValidation(dataset_path, tokenizer,self.image_size)
        self.valid_loader = DataLoader(valid_data, batch_size, shuffle=False)
        return
    

    def prepare_training(self,output_path:str)->None:
        self.output_path=output_path
        os.makedirs(output_path+"/results", exist_ok=True)
        os.makedirs(output_path+"/checkpoints", exist_ok=True)
        return 
    

    def optimizers(self, lr):
        self.opt_vq = torch.optim.Adam(
            list(self.vqgan.encoder.parameters()) +
            list(self.vqgan.decoder.parameters()) +
            list(self.vqgan.codebook.parameters()) +
            list(self.vqgan.pre_quant.parameters()) +
            list(self.vqgan.post_quant.parameters()),
            lr=lr, eps=1e-08, betas=(0.9, 0.999)
        )
        self.opt_disc = torch.optim.Adam(self.discriminator.parameters(),
                                    lr=lr, eps=1e-08, betas=(0.9, 0.999))
        return


    def train_valid(self, epochs, disc_factor, disc_start, perceptual_loss_factor, rec_loss_factor):
        for epoch in range(epochs):
            self.train(self.train_loader, epoch, disc_factor,disc_start,perceptual_loss_factor,rec_loss_factor)
            self.valid(self.valid_loader, epoch, disc_factor,disc_start,perceptual_loss_factor,rec_loss_factor)
        return 


    def train(self, train_loader, epoch, disc_factor,disc_start,perceptual_loss_factor,rec_loss_factor):
        self.vqgan.train()
        self.discriminator.train()
        with tqdm(range(len(train_loader))) as pbar:
            for step, images_text in zip(pbar, train_loader):
                images=images_text[0]
                images = images.to(self.device)
                decoded_images, _, q_loss = self.vqgan(images)

                disc_real = self.discriminator(images)
                disc_fake = self.discriminator(decoded_images)            

                disc_factor = self.vqgan.adopt_weight(disc_factor, epoch*(len(train_loader))+step, threshold=disc_start)

                perceptual_loss = self.perceptual_loss(images, decoded_images)
                rec_loss = torch.abs(images - decoded_images)
                perceptual_rec_loss = perceptual_loss_factor * perceptual_loss + rec_loss_factor * rec_loss
                perceptual_rec_loss = perceptual_rec_loss.mean()
                g_loss = -torch.mean(disc_fake)

                λ = self.vqgan.calculate_lambda(perceptual_rec_loss, g_loss)
                vq_loss = perceptual_rec_loss + q_loss + disc_factor * λ * g_loss

                d_loss_real = torch.mean(F.relu(1. - disc_real))
                d_loss_fake = torch.mean(F.relu(1. + disc_fake))
                gan_loss = disc_factor * 0.5*(d_loss_real + d_loss_fake)

                self.opt_vq.zero_grad()
                vq_loss.backward(retain_graph=True)

                self.opt_disc.zero_grad()
                gan_loss.backward()

                self.opt_vq.step()
                self.opt_disc.step()

                if step % 10 == 0:
                    with torch.no_grad():
                        real_fake_images = torch.cat((images[:4], decoded_images.add(1).mul(0.5)[:4]))
                        vutils.save_image(real_fake_images, os.path.join(self.output_path+"/results", f"{epoch}_{step}.jpg"), nrow=4)

                pbar.set_postfix(
                    VQ_Train_Loss=np.round(vq_loss.cpu().detach().numpy().item(), 5),
                    GAN_Train_Loss=np.round(gan_loss.cpu().detach().numpy().item(), 3)
                )
                pbar.update(0)
            
            torch.save(self.vqgan.state_dict(), os.path.join(self.output_path+"/checkpoints", f"vqgan_epoch_{epoch}.pt"))


    def valid(self, valid_loader, epoch, disc_factor,disc_start,perceptual_loss_factor,rec_loss_factor):
        self.vqgan.eval()
        self.discriminator.eval()
        with torch.no_grad():
            with tqdm(range(len(valid_loader))) as pbar:
                for step, image_text in zip(pbar, valid_loader):
                    images = image_text[0]
                    images = images.to(self.device)
                    decoded_images, _, q_loss = self.vqgan(images)

                    disc_real = self.discriminator(images)
                    disc_fake = self.discriminator(decoded_images)            

                    disc_factor = self.vqgan.adopt_weight(disc_factor, epoch*(len(valid_loader))+step, threshold=disc_start)

                    perceptual_loss = self.perceptual_loss(images, decoded_images)
                    rec_loss = torch.abs(images - decoded_images)
                    perceptual_rec_loss = perceptual_loss_factor * perceptual_loss + rec_loss_factor * rec_loss
                    perceptual_rec_loss = perceptual_rec_loss.mean()
                    g_loss = -torch.mean(disc_fake)

                    #how to work with the loss function in validaiton step?
                    #λ = self.vqgan.calculate_lambda(perceptual_rec_loss, g_loss)
                    vq_loss = perceptual_rec_loss + q_loss + disc_factor * g_loss

                    d_loss_real = torch.mean(F.relu(1. - disc_real))
                    d_loss_fake = torch.mean(F.relu(1. + disc_fake))
                    gan_loss = disc_factor * 0.5 * (d_loss_real + d_loss_fake)

                    pbar.set_postfix(
                        VQ_Valid_Loss=np.round(vq_loss.cpu().detach().numpy().item(), 5),
                        GAN_Valid_Loss=np.round(gan_loss.cpu().detach().numpy().item(), 3)
                    )
                    pbar.update(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="stage1: vit-vqgan")
    args = parser.parse_args()

    #set up device
    args.device="cpu"
    args.output_path="/assets"

    #VitEncoder and VitDecoder
    args.image_size=(64,64) #(256,256)
    args.image_channels=3 #3
    args.patch_size=(8,8) #(8,8)
    args.dim_model=32 #512
    args.nhead=2 #8
    args.nlayer=2 #8
    args.dim_feedforward=4*args.dim_model #2048
    

    #Codebook
    args.codebook_vocab_size=128 #8192
    args.latent_dim=args.dim_model 
    args.beta=0.25


    #Build Model
    stage1=Stage1(args.image_size, 
                  args.image_channels, 
                  args.patch_size, 
                  args.dim_model, 
                  args.nhead, 
                  args.nlayer, 
                  args.dim_feedforward, 
                  args.codebook_vocab_size, 
                  args.latent_dim, 
                  args.beta, 
                  args.device)
    
    #Make directory
    stage1.prepare_training(args.output_path)
    
    #Load Dataset
    args.dataset_path=r"/assets/flowers"
    args.batch_size=2
    args.text_seq_len=128
    stage1.load_dataset(args.dataset_path, args.batch_size, args.text_seq_len)
    

    #Optimizer
    args.learning_rate=0.001
    stage1.optimizers(args.learning_rate)


    #Train and Valid
    args.epochs=3
    args.disc_factor=1.0
    args.disc_start=2
    args.perceptual_loss_factor=1.0
    args.rec_loss_factor=1.0

    stage1.train_valid(args.epochs, args.disc_factor, args.disc_start, args.perceptual_loss_factor, args.rec_loss_factor)