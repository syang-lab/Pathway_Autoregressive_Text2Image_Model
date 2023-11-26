## Pathway Autoregressive Text to Image Model (Parti)
Unofficial pytorch implementation of Google's paper: scaling autoregressive models for content-rich text-to-image generation. 

## Simplified Two Stage Model Explaination 
Stage1: ViT-VQGAN

intuitive explanation: learn an image codebook (index/token, vectors) through image encoding and decoding images into quantized components.

ViT encoder: encode images into quantized embedding space

* input: images [batch_size, channels, H, W] 
  
* output: encoded images [batch_size, grid_h*grid_w, latent_dim]

ViT decoder: decode vectors from embedding space to images

* input: encoded images [batch_size, grid_h*grid_w, latent_dim]
  
* output: images  [batch_size, channels, H, W] 

Codebook: embedding space lookup table
  
* input: encoded images [batch_size, grid_h*grid_w, latent_dim]
  
* output: image tokens [batch_size, grid_h*grid_w]

Loss function: 
* reconstruction loss
* discriminator loss
* preceptual loss
* codebook loss: gradient flow directly from decoder to encoder, because the quantization step is not differentiable.

Stage2: Transformer (encoder and decoder)

intuitive explanation: tokenized text, while tokenized images according to the VQGAN codebook. Then, text tokens pass through the transformer encoder, and image tokens pass through the transformer decoder and output the predicted image tokens. The predicted image tokens are converted to vectors according to the codebook, which are eventually decoded into images via the Vit decoder.

* encode and tokenized images: Vit encoder+codebook

* input to transformer : tokenized images [batch_size, grid_h*grid_w (image sequence length)], tokenized text [batch_size, text sequence length]

* output from tranformer : image tokens [batch_size, (image sequence length)]
  
* decode tokens and generate images: Vit decoder+codebook 

Loss function: 
* cross entropy 

## Versions
Parti_pytorch_V5: Create the model in a way that is clear and easy to understand, facilitating user comprehension and execution.
Parti_pytorch_V6: Establish compatibility with the pre-trained VQGAN model accessible on GitHub: https://github.com/thuanz123/enhancing-transformers, enabling seamless reuse of the retained VQGAN model for the parti mode.

## Todo 
- [x] implement pytorch-light to simplify training and validation.
- [x] setup config files.
- [ ] track experiments with w&b
- [ ] train transformer with existing VQGAN pre-trained models.


## Citations
This repository modifies VQGAN code from following repos:
```bibtex
@misc{enhancing-transformers,
  url = {https://github.com/thuanz123/enhancing-transformers},
  year = {2023}
}
```
```bibtex
@misc{VQGAN-pytorch,
  url = {https://github.com/dome272/VQGAN-pytorch/tree/main},
  year = {2023}
}
```
This repository inspired by the following papers:
```bibtex
@article{yu2022scaling,
  title={Scaling autoregressive models for content-rich text-to-image generation},
  author={Yu, Jiahui and Xu, Yuanzhong and Koh, Jing Yu and Luong, Thang and Baid, Gunjan and Wang, Zirui and Vasudevan, Vijay and Ku, Alexander and Yang, Yinfei and Ayan, Burcu Karagol and others},
  journal={arXiv preprint arXiv:2206.10789},
  volume={2},
  number={3},
  pages={5},
  year={2022},
  publisher={Jun}
}
```
```bibtex
@inproceedings{esser2021taming,
  title={Taming transformers for high-resolution image synthesis},
  author={Esser, Patrick and Rombach, Robin and Ommer, Bjorn},
  booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
  pages={12873--12883},
  year={2021}
}
```
```bibtex
@article{yu2021vector,
  title={Vector-quantized image modeling with improved vqgan},
  author={Yu, Jiahui and Li, Xin and Koh, Jing Yu and Zhang, Han and Pang, Ruoming and Qin, James and Ku, Alexander and Xu, Yuanzhong and Baldridge, Jason and Wu, Yonghui},
  journal={arXiv preprint arXiv:2110.04627},
  year={2021}
}
```



