# vocal-remover

This is a deep-learning-based tool to extract instrumental track from your songs.

Ok, some updates on the off chance anyone is reading anything from my fork.

I have compared three variants of the frame primer and have converged on the final architecture. The three variants tested out were: 1) frame primer without any downsampling - just a stack of frame primer encoders 2) unet with frame primer encoders/decoders at every resolution 3) unet with frame primer encoder modules in the bridge of the unet as with TransUNet. From my tests, 2 appears to work best, 1 works least of all, and 3 works fairly well. The skip attention seems to be what helps with the unet variant, as using only frame primer decoders yields best results. I am still carrying out tests to see whether or not deper unets work best with this; current signs point toward this being the case, however it could also just be random initializations having vastly different outcomes so I am not entirely sure right now; should have a better view of that soon.

Previous unmasking pretraining that was done in the style of BERT didn't seem to yield great results, both due to simplified architecture used and the unmasking task not being ideal for computer vision/audition pretraining, and the DDeP paper mentions this not being a very strong task for pretraining for CV/A. I am currently preparing to test a pretraining run on 75+ days of music using the pretraining routine described in "Decoder Denoising Pretraining for Semantic Segmentation." frame_primer/dataset_denoising.py contains the denoising dataset which as stated in DDeP can be viewed as a single step diffusion process. The routine described in the paper is what is currently coded; the frame primer is predicting the noise rather than predicting the denoised representation which according to the paper is supposed to yield better results. As in the paper, there is a scaling parameter gamma, and the training example is defined as X' = sqrt(gamma) * X + sqrt(1 - gamma) * Y where Y = N(0, sigma) and is the target for the L1 loss.

Below is a diagram of the frame primer:
![image](https://user-images.githubusercontent.com/30326384/180595547-f60b055a-d49e-4861-b126-581ee5165f02.png)

## References
- [1] Jansson et al., "Singing Voice Separation with Deep U-Net Convolutional Networks", https://ismir2017.smcnus.org/wp-content/uploads/2017/10/171_Paper.pdf
- [2] Takahashi et al., "Multi-scale Multi-band DenseNets for Audio Source Separation", https://arxiv.org/pdf/1706.09588.pdf
- [3] Takahashi et al., "MMDENSELSTM: AN EFFICIENT COMBINATION OF CONVOLUTIONAL AND RECURRENT NEURAL NETWORKS FOR AUDIO SOURCE SEPARATION", https://arxiv.org/pdf/1805.02410.pdf
- [4] Liutkus et al., "The 2016 Signal Separation Evaluation Campaign", Latent Variable Analysis and Signal Separation - 12th International Conference
- [5] Vaswani et al., "Attention Is All You Need", https://arxiv.org/pdf/1706.03762.pdf
- [6] So et al., "Primer: Searching for Efficient Transformers for Language Modeling", https://arxiv.org/pdf/2109.08668v2.pdf
- [7] Huang et al., "Music Transformer: Generating Music with Long-Term Structure", https://arxiv.org/pdf/1809.04281.pdf
- [8] He et al., "RealFormer: Transformer Likes Residual Attention", https://arxiv.org/pdf/2012.11747.pdf
- [9] Asiedu et all., "Decoder Denoising Pretraining for Semantic Segmentation", https://arxiv.org/abs/2205.11423
