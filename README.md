# vocal-remover

This is a deep-learning-based tool to extract instrumental track from your songs.

This is basically a junk personal fork, I will make a new fork when things are finished that are cleaner and more protected. Pretraining is for the most part done with the primer architecture, however I am still in the process of finetuning it. Because I was getting anxious and wanted to work on something new, for the next day or so I'm shifting to a new architecture; initial tests seem to point toward this being the best validation loss yet, but need to take it further to be able to make any reasonable assertion. Diagram of architecture is below, code is in frame_primer/frame_resnet.py.

Made some further changes after reading the "Decoder Denoising Pretraining for Semantic Segmentation" paper and after recent successes of diffusion models wished to give it a try for pretraining the entire network (also because they call out unmasking as not as strong for computer vision which this still pretty much is). As with DDeP, the pretraining is trying to predict the noise signal rather than the denoised image. Need to test how it transfers to the downstream task of vocal removing, but if all goes well I'll try doing pretraining on a smaller sequence length of ~10 seconds of audio first then move up to 40 seconds eventually. Currently testing out a new variant of the frame primer; this one is a standard u-net, and at the middle there is a transformer as in the TransUNet. Really the only difference with TransUnet and this now is the use of the frame convolutions, relative positional encoding, and multi-dconv-head attention along with the rectified relu activation in the transformer module and of course how the 'patches' are formed (here you can consider each frame as a patch really; currently I have it set up to dedicate a head to each channel so with 5 layers @ 2 channels that would be 10 heads of attention). In previous versions adding more channels didn't seem to help a whole lot; have tested a version with twice as many unet channels but it didn't do super well; it seems that the dense net bottleneck version is the better variant.

Had a model previously pretraining but will probably not be releasing it as it is fairly useless. While it converges rapidly for vocal removing, I wasn't able to find a good way to finetune it such that it really warranted the amount of compute I sunk into it. I am now shifting to a denoising pretraining setup as described in "Decoder Denoising Pretraining for Semantic Segmentation" to see how that works given the similarity between architectures used and downstream task.

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
