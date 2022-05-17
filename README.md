# vocal-remover

This is a deep-learning-based tool to extract instrumental track from your songs.

This is an experimental fork that is changing rapidly. It's main focus is to bring both transformers and pretraining to the problem of track separation, specifically vocal removing. The current architecture used is in evl_frame_transformer.py. This consists of a DenseNet where each block is a frame transformer encoder. The frame transformer encoder consists of a 1x1 kernel bottleneck to 1 channel followed by the evolved transformer encoder architecture with the modifications from Primer-EZ with an attention mechanism that utilizes relative positional encoding as seen in the Music Transformer. The final layer is a 1x1 kernel convolution that bottlenecks the input and all output transformer encoder outputs into a stereo channel output.

Currently, train-pre.py will launch a pre-training script that follows BERT's pre-training. It takes as input a spectrogram of cropsize + next_frame_chunk_size frames. The default settings split evenly with 552 frames for spectrogram a and spectrogram b. Both spectrograms (or half the time, one long spectrogram) will have frames randomly whited out by default at a 20% chance (whiteed out frames are treated as mask token so that sigmoid can be used to 'sculpt' the correct frame; hopefully this will end up being more relevant to vocal removing as well but that is unfounded). As with bert, 10% of the time these selected frames are randomized (however this adds noise rather than completely randomize, I guess you could consider this as having it learn to denoise along with unmasking) and another 10% of the time they are left alone. The neural network then will output a single scalar specifying whether or not the second part of the spectrogram is a continuation of the first part of the spectrogram in addition to its unmasked prediction output.

I will be releasing a pretrained checkpoint in the coming weeks. I have a dataset that consists of 78,046 spectrograms, each spectrogram having 2048 frames. Currently it is trained on just instrumental songs and mix songs, for finetuning I will be using my vocal stem database of 492 vocal tracks to create exponentially more data for the downstream task. I am going to be adding significantly more music to my pre-training dataset within the week.

If someone would like to work with me please reach out to me at carperbr@gmail.com.

## References
- [1] Jansson et al., "Singing Voice Separation with Deep U-Net Convolutional Networks", https://ismir2017.smcnus.org/wp-content/uploads/2017/10/171_Paper.pdf
- [2] Takahashi et al., "Multi-scale Multi-band DenseNets for Audio Source Separation", https://arxiv.org/pdf/1706.09588.pdf
- [3] Takahashi et al., "MMDENSELSTM: AN EFFICIENT COMBINATION OF CONVOLUTIONAL AND RECURRENT NEURAL NETWORKS FOR AUDIO SOURCE SEPARATION", https://arxiv.org/pdf/1805.02410.pdf
- [4] Liutkus et al., "The 2016 Signal Separation Evaluation Campaign", Latent Variable Analysis and Signal Separation - 12th International Conference
- [5] So et al., "The Evolved Transformer", https://arxiv.org/pdf/1901.11117.pdf
- [6] Huang et al., "Music Transformer: Generating Music with Long-Term Structure", https://arxiv.org/pdf/1809.04281.pdf
- [6] So et al., "Primer: Searching for Efficient Transformers for Language Modeling", https://arxiv.org/pdf/2109.08668v2.pdf
- [7] Devlin et al., "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding", https://arxiv.org/pdf/1810.04805.pdf