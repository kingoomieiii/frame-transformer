# vocal-remover

This is a deep-learning-based tool to extract instrumental track from your songs.

For the most part, this fork has converged on a final architecture that seems to have the most benefits. The current architecture is called a frame primer. It consists of a residual u-net modified to use frame convolutions, frame primer encoders and frame primer decoders. After each encoder in the u-net there is a sequence of densely connected frame primer encoders. Before each decoder, there is a sequence of densely connected frame primer decoders. All of the following files are located in the frame_primer folder. I am currently pretraining a model with 121,503,734 parameters on 61+ days of music. Each spectrogram in my dataset captures around 40 seconds of audio (2048 fft 1024 hl 2048 cropsize), and it is learning to unmask chunks of 64 frames with a mask rate of 0.2.

Update: Testing out pretraining at 250k optimization steps has had actually somewhat surprising results; it allows for extremely rapid convergence with the vocal remover even with some pretty glaring issues that were overlooked (dropout rate of 50% on inner decoders, nearest neighbor upsampling, and residual attention layer not being used). I also have made some changes to try to make the architecture more amenable to mixed precision, but I am not entirely sure if it works just yet - full precision works without issue however and requires no gradient clipping. If mixed precision still fails, I will likely change the code to alternate between full and mixed precision on the fly in order to mitigate the issues; the downside here is that you don't really gain any reward in terms of VRAM that way, but mixed precision is faster on an rtx 3080 ti at least so there is a benefit there. I am currently testing a decoder only variant which is performing quite well, will allow this to continue training and will upload checkpoints hopefully in a week or so.

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
