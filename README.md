# vocal-remover

This is a deep-learning-based tool to extract instrumental track from your songs.

For the most part, this fork has converged on a final architecture that seems to have the most benefits. The current architecture is called a frame primer. It consists of a residual u-net modified to use frame convolutions, frame primer encoders and frame primer decoders. After each encoder in the u-net there is a sequence of densely connected frame primer encoders. Before each decoder, there is a sequence of densely connected frame primer decoders. All of the following files are located in the frame_primer folder. I am currently pretraining a model with 121,503,734 parameters on 61+ days of music. Each spectrogram in my dataset captures around 40 seconds of audio (2048 fft 1024 hl 2048 cropsize), and it is learning to unmask chunks of 64 frames with a mask rate of 0.2.

Currently finalizing a pretraining run that is at ~~314,877~~ 633,000 optimization steps - I'm guessing it'll be finished training by Saturday; I have it currently pretraining on a cropsize of 512 (technically I did two epochs of 1024 but shifted back to 512, will retrain the longer sequence run again but didn't want to throw any  optimization steps out), will be shifting to 1024 for about 100k steps and then will do at least 100k steps on 2048 which should give a pretty similar breakup compared to BERT. My intention is to allow it to train at a cropsize of 512 for six more epochs which will bring the total optimization steps to 872,693 optimization steps. From here, I will increase cropsize to 1024 and train for two more epochs adding 139,454 more optimization steps to bring it to 1,012,147 optimization steps. From here, I will train at the largest cropsize of 2048 for two epochs to finalize the model and allow it to learn positional embeddings as they mention in the BERT paper. This process should be complete by Thursday afternoon, after which I will begin the process of finetuning the model. I will likely upload the pretrained checkpoint on Thursday as well. Currently the model is at 150,617,664 parameters.

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
