# vocal-remover

This is a deep-learning-based tool to extract instrumental track from your songs.

This is a bunch of experimental code right now. This is not in a usable state, but hopefully will be soon. Basically just a transformer sandbox for audio at this point. Main focus at this very moment is coming up with an effective pretraining routine for the transformer (although it would probably benefit the convolutional variant as well)

This architecture started off initially as a variation of tsurumeso's vocal remover but has evolved (no pun intended) to a fairly standalone architecture. At this point, the only real similarity is the bottleneck that is used before the transformer blocks. The architecture is a cross between the Evolved Transformer, the Music Transformer, and the Primer with the bottlenecking technique applied in MMDENSELSTM. This no longer downsamples the input and processes the input spectrogram directly. The transformer encoders are set up in a dense-net style where each transformer encoder is passed to the following encoders by concatenating its output to the input. The bottleneck that is present in each transformer then is responsible for extracting relevant information in the form of its in projection. My current plan is to pretrain this model in a Bert self-supervised style pretraining setup, lock all but the final linear layer (will also test decaying learning rates for each layer), and then attempt to finetune on removing vocals from there. This gives the added benefit of having an audio modeling pretrained checkpoint that can be used for other tasks as well.

I will release checkpoints eventually, but I keep iterating on this daily and making progress which makes it difficult for me to stop since I'm not a researcher trying to publish a paper - I am approaching this as an engineer rather than as a researcher clearly and making progress has become addicting to me. My plan is to pretrain the autoregressive variant to predict spectrogram magnitudes and then finetune on vocal removing. I am currently in the process of building a local compute cluster with each node having an RTX 3090 Ti, but currently I only have a single RTX 3090 Ti and a single RTX 3080 Ti (will be a months long endeavor, however I should be able to get a third node setup within a month at most I'd say). Eventually I will be uploading custom cloud software to my github account and who knows maybe start some cheap cloud service at some point for a limited number of people to try and help others with their work as I understand how unbelievably frustrating it is to wait hours for the results of a single experiment. 

If someone would like to work with me please reach out to me at carperbr@gmail.com.

Example of vocal extraction at only the second epoch which shows how it handles low end. 
![image](https://user-images.githubusercontent.com/30326384/167711869-e02c7a4a-8baf-4119-a531-232836e93187.png)

## References
- [1] Jansson et al., "Singing Voice Separation with Deep U-Net Convolutional Networks", https://ismir2017.smcnus.org/wp-content/uploads/2017/10/171_Paper.pdf
- [2] Takahashi et al., "Multi-scale Multi-band DenseNets for Audio Source Separation", https://arxiv.org/pdf/1706.09588.pdf
- [3] Takahashi et al., "MMDENSELSTM: AN EFFICIENT COMBINATION OF CONVOLUTIONAL AND RECURRENT NEURAL NETWORKS FOR AUDIO SOURCE SEPARATION", https://arxiv.org/pdf/1805.02410.pdf
- [4] Liutkus et al., "The 2016 Signal Separation Evaluation Campaign", Latent Variable Analysis and Signal Separation - 12th International Conference
- [5] So et al., "The Evolved Transformer", https://arxiv.org/pdf/1901.11117.pdf
- [6] Huang et al., "Music Transformer: Generating Music with Long-Term Structure", https://arxiv.org/pdf/1809.04281.pdf
- [6] So et al., "Primer: Searching for Efficient Transformers for Language Modeling", https://arxiv.org/abs/2109.08668v2
