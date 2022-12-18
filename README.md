# frame-transformer

This fork is mainly a research fork and will change frequently and there is like zeo focus on being user friendly. I am happy to answer any and all questions however.

First 'fully trained' checkpoint is here: https://mega.nz/file/3ggnHC6Y#mZl-gjUSKpvstuGZrUQmOxM-QyqCxrUgjthBrHIU0gc. This is at about 1 million optimization steps, little bit more. Still doing some testing to see how best to inference with it, but a cropsize of 2048 and padding of 0 seems to do well. A cropsize of 1024 and padding of 512 will likely perform quite well. I will begin training larger versions of this model now, however this checkpoint should work on most GPUs as it is far less intense compared to the full frame transformers. I have also begun experimenting with an entirely new neural network that no longer uses a u-net and relies instead on residual vector quantization coupled with a multichannel transformer as opposed to a decoder as in the current version. I'm noticing some issues with fretless bass in some bands so I need to do some more work clearly. 

I will be moving the transformer modules from the current model at the 87th 'epoch' into a larger convolutional backbone and retrain on my dataset to see how that effects things, however number of attention maps seems to be more important than number of feature maps. My hope is that the thin frame transformer will be conducive to transfer learning and allow to bootstrap larger models more quickly.

Will have new checkpoints soon and will update readme then, first full training session for the frame transformer is almost complete. The single attention map frame transformer is currently at around 800k optimization steps and has increased its context to a cropsize of 1280 at this point. I hvae been using a cropsize of 2048 for validation loss and with each increase in resolution validation loss on the larger context drops quite drastically which is a pretty strong indicator that the transformer modules are playing a key role in the neural network. Will upload at 1 million optimization steps, after that I'm going to focus on training a full frame transformer as testing does show that more attenion maps leads to better results (and also way more memory consumption...) 

This project consists of two main pieces: a new neural network that I have created with inspiration from various transformer papers and tsurumeso's original MMDENSELSTM implementation (the idea of compressing the feature maps into a single channel and processing the frames with an LSTM is the only reason this fork exists). The second piece seems to be the more critical piece, and it is a dataloader/technique that I refer to as voxaug. This is an augmentation technique, or more accurately a data synthesis technique, which mixes instrumental songs with vocal stems to create vast amounts of data. My current dataset consists of around 15k instrumental songs and 1500 vocal stems which means there are about 22,500,000 song pairs and far more spectrogram training examples. This dataloader appears to be key in allowing the frame transformer to extrapolate.

Currently training the model in frame_tranformer_thin.py. This variant uses the single channel technique seen in the original fork and then uses a single channel transformer encoder for the encoder sequence in the u-net and single channel transformer decoders for the decoding sequence; it uses squared ReLU as in the Primer paper however the attention mechanism does not include convolutions given the convolutional backbone for the transformer. So far it seems to have surpassed all my previous versions at only 200k optimization steps. After this one is finished, I will train a larger model using the full frame transformer setup. I expect it to perform a bit better, but it will also be far slower so there are trade-offs there (this one can also have more context which should help). Here is a checkpoint for the thin frame transformer at epoch 6. This is trained on about 40-45 days of instrumental music that is randomly mixed with 1500 vocal tracks; this uses the voxaug2 dataset which ensures that no training example is repeated until every combiniation of training data has been seen which means that this has never seen the same training example more than once. This checkpoint uses the default hyperparameters in inference_thin.py. https://mega.nz/file/rw4RkBrD#2MbnMFujw8hGqsTZxaW16FWwi-m-GW4vqtask45wJBM

## References
- [1] Jansson et al., "Singing Voice Separation with Deep U-Net Convolutional Networks", https://ismir2017.smcnus.org/wp-content/uploads/2017/10/171_Paper.pdf
- [2] Takahashi et al., "Multi-scale Multi-band DenseNets for Audio Source Separation", https://arxiv.org/pdf/1706.09588.pdf
- [3] Takahashi et al., "MMDENSELSTM: AN EFFICIENT COMBINATION OF CONVOLUTIONAL AND RECURRENT NEURAL NETWORKS FOR AUDIO SOURCE SEPARATION", https://arxiv.org/pdf/1805.02410.pdf
- [4] Liutkus et al., "The 2016 Signal Separation Evaluation Campaign", Latent Variable Analysis and Signal Separation - 12th International Conference
- [5] Vaswani et al., "Attention Is All You Need", https://arxiv.org/pdf/1706.03762.pdf
- [6] So et al., "Primer: Searching for Efficient Transformers for Language Modeling", https://arxiv.org/pdf/2109.08668v2.pdf
- [7] Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding", https://arxiv.org/abs/2104.09864
- [9] Asiedu et all., "Decoder Denoising Pretraining for Semantic Segmentation", https://arxiv.org/abs/2205.11423
