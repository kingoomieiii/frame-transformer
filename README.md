# vocal-remover

This is a deep-learning-based tool to extract instrumental track from your songs.

This is a bunch of experimental code right now. This is not in a usable state, but hopefully will be soon. Basically just a transformer sandbox for audio at this point. Main focus at this very moment is coming up with an effective pretraining routine for the transformer (although it would probably benefit the convolutional variant as well).

This architecture is what I call a frame transformer. It consists of a series of transformer blocks that use the same bottlenecking concept as in MMDENSELSTM in order to extract a representation from the input channels. The transformer modules form a DenseNet with the input, and the final output linear layer bottlenecks them back to the original stereo channel input. The transformer blocks themselves consist of the aforementioend linear bottleneck to extract the single channel representation. After this, it consists of a pre-norm evolved transformer encoder with the additions from the Primer-EZ architecture. The attention used is called multiband frame attention, however it is effectively just multihead attention where the frequency bins are split up across the various heads. The multiband frame attention module uses relative positional encoding as seen in the Music Transformer.

Right now my focus is on train-pre.py, so train.py probably isn't going to work if you try it (though it wouldn't be that much effort to fix it). train-pre.py currently launches a pretraining regimen that follows Bert's pretraining. It takes as input a spectrogram of cropsize + next_frame_chunk_size frames. The default settings split evenly with 552 frames for spectrogram a and spectrogram b. Both spectrograms (or half the time, one long spectrogram) will have frames randomly whited out by default at a 20% chance. As with bert, 10% of the time these selected frames are randomized and another 10% of the time they are left alone. The neural network then will output a single scalar specifying whether or not the second part of the spectrogram is a continuation of the first part of the spectrogram.

I will be releasing a pretrained checkpoint in the coming weeks. I have a dataset that consists of many thousands of songs, I've honestly lost track at this point. 

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
- [7] Devlin et al., "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding", https://arxiv.org/pdf/1810.04805.pdf