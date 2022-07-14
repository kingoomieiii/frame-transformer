# vocal-remover

This is a deep-learning-based tool to extract instrumental track from your songs.

This is basically a junk personal fork, I will make a new fork when things are finished that are cleaner and more protected. Pretraining is for the most part done with the primer architecture, however I am still in the process of finetuning it. Because I was getting anxious and wanted to work on something new, for the next day or so I'm shifting to a new architecture; initial tests seem to point toward this being the best validation loss yet, but need to take it further to be able to make any reasonable assertion. Diagram of architecture is below, code is in frame_primer/frame_resnet.py.

Currently experimenting with a new architecture that appears to be working quite well: 
![image](https://user-images.githubusercontent.com/30326384/179087464-d90206dd-9302-4c44-a4e2-83d3967e9ff7.png)


For the most part, this fork has converged on a final architecture that seems to have the most benefits. The current architecture is called a frame primer. It consists of a residual u-net modified to use frame convolutions, frame primer encoders and frame primer decoders. After each encoder in the u-net there is a sequence of densely connected frame primer encoders. Before each decoder, there is a sequence of densely connected frame primer decoders. All of the following files are located in the frame_primer folder. I am currently pretraining a model with 121,503,734 parameters on 61+ days of music. Each spectrogram in my dataset captures around 40 seconds of audio (2048 fft 1024 hl 2048 cropsize), and it is learning to unmask chunks of 64 frames with a mask rate of 0.2.

Currently finalizing a pretraining run of a model with 150,617,664 parameters on ~~64.5~~ 70.6 days of music that is at ~~314,877 633,000 946,000~~ 1,001,000 optimization steps :D - will be training for at least 12 more hours, probably another 16 after that for a second epoch at the full sequence length of ~40 seconds. It is now training on a cropsize of 2048 after quite a few optimization steps at 512 and a few at 1024. My goal is to bring it up 1,206,497 steps. Next steps are finetuning on the vocal remover dataset, will do some experimenting with various methods of finetuning (LLRD, only training transformer modules, only training last N decoders, etc). After I finetune I will upload the finetuned model and the pretrained model. I will probably train another model to do some other task, likely some form of mix balancing tool.

Below is a summary of modules and training scripts:

FramePrimer - this module is the base network. It consists of four submodules: a sequence of FrameEncoders, a sequence of FramePrimerEncoders, a sequence of FrameDecoders and one conv2d, and a sequence of FramePrimerDecoders.

FrameConv - this is a variation of 2d conv blocks that I made to process information without respect to neighboring frames. It uses Nx1 convolutions rather than NxN or NxM. It is only able to convolve features from the same frame. For normalization, it uses nn.InstanceNorm2d, however the channel and frame dimensions are transposed in order to normalize each frames embeddings similar to layer norm but without element-wise parameters.

ResBlock - This is a standard res block but using frame convolutions.

FrameEncoder - this module is an encoder for individual frames of a spectrogram. It consists of a ResBlock module and is responsible for downsampling.

FrameDecoder - this module is a decoder for individual frames of a spectrogram. It consists of a ResBlock module and a nn.Upsample module if needed.

FramePrimerEncoder - this module is a variation of the primer encoder architecture. It first bottlenecks the input to a single channel; from here, it is a standard primer encoder with relative positional encoding.

FramePrimerDecoder - this module is a variation of the primer decoder architecture. It first bottlenecks the input to a single channel; from here, it is a standard primer decoder with relative positional encoding. The memory for this decoder is the skip connection in the u-net, which allows for each frame to query for global information from the skip connections.

Right now I'm mainly only focusing on train-pre.py and train.py. train-gan.py works, but it seems it will require quite a bit of training. I will likely modify this script to be for vocal removing rather than pretraining.

train-pre.py - This is pretraining that makes use of L1 reconstruction on unmasked regions; it only considers loss for the unmasked regions as in HuBERT. Unmasking frequency bins is done by setting all frequency bins within a token to 1; this allows the model to learn a multiplicatve mask with which to unmask the audio which helps with convergence.

train.py - This is the original training script with some modifications (learning rate warmup w/ polynomial decay, mixed precision, wandb).

## References
- [1] Jansson et al., "Singing Voice Separation with Deep U-Net Convolutional Networks", https://ismir2017.smcnus.org/wp-content/uploads/2017/10/171_Paper.pdf
- [2] Takahashi et al., "Multi-scale Multi-band DenseNets for Audio Source Separation", https://arxiv.org/pdf/1706.09588.pdf
- [3] Takahashi et al., "MMDENSELSTM: AN EFFICIENT COMBINATION OF CONVOLUTIONAL AND RECURRENT NEURAL NETWORKS FOR AUDIO SOURCE SEPARATION", https://arxiv.org/pdf/1805.02410.pdf
- [4] Liutkus et al., "The 2016 Signal Separation Evaluation Campaign", Latent Variable Analysis and Signal Separation - 12th International Conference
- [5] Vaswani et al., "Attention Is All You Need", https://arxiv.org/pdf/1706.03762.pdf
- [6] So et al., "Primer: Searching for Efficient Transformers for Language Modeling", https://arxiv.org/pdf/2109.08668v2.pdf
- [7] Huang et al., "Music Transformer: Generating Music with Long-Term Structure", https://arxiv.org/pdf/1809.04281.pdf
- [8] He et al., "RealFormer: Transformer Likes Residual Attention", https://arxiv.org/pdf/2012.11747.pdf
- [9] Devlin et al., "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding", https://arxiv.org/pdf/1810.04805.pdf
- [10] Hsu et al., "HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units", https://arxiv.org/pdf/2106.07447.pdf
