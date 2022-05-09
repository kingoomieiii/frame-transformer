# vocal-remover

This is a deep-learning-based tool to extract instrumental track from your songs.

This is a variation of tsurumeso's vocal remover that I've been working on for a while. I have put quite a bit of effort into converging on a solution that was able to be applied to metal music with a higher degree of accuracy, which consequently led to improvements in other areas. This uses a custom architecture that is a hybrid between a convolutional neural network, the evolved transformer, the music transformer, the primer, and u-net architectures that is obviously heavily inspired by MMDENSELSTM. The convolutional portion uses Nx1 kernels to convolve only features of the same frame. This works quite well. This architecture handles low end far better than previous versions I've tested (at least on metal), and volume dropout is nearly non-existent in this version (again on metal).

All I ask is that if you make use of what I've done you at least credit me in some way, especially if the ideas are used in an academic setting however farfetched that might be. I am an engineer trying to make a name for myself especially when it comes to AI, and while I work for a fairly large company we don't do much with machine learning. This architecture was heavily inspired by MMDENSELSTM and Tsurumeso's repo, however does have quite a few key differences (namely the frame convolutions and the use of transformer modules that I  coded from scratch in order to learn how they work more intimately). The results are quite promising even when using a measely 16 channels at the widest part of the convolutional portion of the network and increasing the channel count actually doesn't even seem to help much.

I may or may not release checkpoints. My dataset is very large (49,257 spectrograms of 2048 frames each for instrumentals alone, 2,884 vocal spectrograms of 2048 frames each w/ hop length of 1024 which gives 142,057,188 training examples before any further augmentation) and thus can reach high levels of quality, however I am not convinced that it wouldn't just be stolen and have someone claim they trained it at this point. Starting to feel a bit jaded...

If someone would like to work with me please reach out to me at carperbr@gmail.com.

Example of vocal extraction which shows how well it handles low end:
![image](https://user-images.githubusercontent.com/30326384/167472544-8bacf9f4-3155-4ff7-9716-7a8e06d5bb70.png)

## References
- [1] Jansson et al., "Singing Voice Separation with Deep U-Net Convolutional Networks", https://ismir2017.smcnus.org/wp-content/uploads/2017/10/171_Paper.pdf
- [2] Takahashi et al., "Multi-scale Multi-band DenseNets for Audio Source Separation", https://arxiv.org/pdf/1706.09588.pdf
- [3] Takahashi et al., "MMDENSELSTM: AN EFFICIENT COMBINATION OF CONVOLUTIONAL AND RECURRENT NEURAL NETWORKS FOR AUDIO SOURCE SEPARATION", https://arxiv.org/pdf/1805.02410.pdf
- [4] Liutkus et al., "The 2016 Signal Separation Evaluation Campaign", Latent Variable Analysis and Signal Separation - 12th International Conference
- [5] So et al., "The Evolved Transformer", https://arxiv.org/pdf/1901.11117.pdf
- [6] Huang et al., "Music Transformer: Generating Music with Long-Term Structure", https://arxiv.org/pdf/1809.04281.pdf
- [6] So et al., "Primer: Searching for Efficient Transformers for Language Modeling", https://arxiv.org/abs/2109.08668v2
