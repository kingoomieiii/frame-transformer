# frame-transformer

This is a collection of tools for working with spectrograms using transformers. It includes a vocal remover that has evolved from tsurumeso's vocal remover, a phase predictor, pretraining, and a vector quantizer for frames. Right now I am working on training the vocal remover, however I will be moving onto these others after that is complete.

I'm calling this architecture a frame transformer. It consists of a position-wise linear residual u-net with a multichannel transformer. Currently training a version of this out, it is learning way faster than I would have thought so I'm pretty excited. It is able to keep up with the convolutional variant with half the context and struggles less with stuff like fretless bass. 

This fork also makes use of a dataset I refer to as voxaug in order to satisfy the transformers need for large amounts of data. This dataset randomly selects from a library of instrumental music and a library of vocal tracks and mixes them together for the neural network to train on. This has the benefit of inflating data exponentially as well as ensuring data is perfect for the removal process. To an extent you could view this as self-supervised learning in that its learning to remove a mask of vocals. My instrumental dataset consists of 30.88 days worth of music while my vocal stem library consists of 1416 full song vocal tracks. I will be uploading checkpoints for a 357,493,618 parameter model after it trains for a few days.

## Module Descriptions ##

* **FrameTransformer** - The core of the neural network. This consists of a series of encoders and decoders, with encoders defined as frame_transformer_encoder(frame_encoder(x)) and decoders defined as frame_transformer_decoder(frame_decoder(x, skip)). It also includes an output depthwise linear layer in the form of a weight matrix.

* **MultichannelLinear** - This was my solution to having parallel linear layers. Instead of having individual linear layers, I compressed them into a single weight matrix with a channel dimension and make use of batched matrix multiplication. It also includes a depth-wise linear layer for increasing channel count (compression of frequency axis and expansion of channels is still necessary for this to learn well, although it seems to have less of a reliance on channels than a convolutional neural network).

* **FrameNorm** - Just a helper module; applies layer norm to each frame of each channel with shared affine transform.

* **FrameEncoder** - position-wise encoder for each frame responsible for downsampling and expansion of channels. This consists of a residual block made from multichannel linear layers to allow for each channel to learn its own position-wise linear layer. It takes inspiration from the transformer architecture and uses residual blocks in that style - linear2(activation(linear1(norm(x)))).

* **FrameDecoder** - position-wise decoder for each frame responsible for upsampling and reduction of channels. This consists of two residual blocks; the first allows each channel to learn its own position-wise and depth-wise residual block for upsampling frames, and the second residual block integrates the skip connection by concatenating it with the output of the first block and reducing it back to out_channels with a position-wise and depth-wise multichannel linear layer.

* **MultichannelMultiheadAttention** - This module is an extension of multihead attention for multiple channels. Each channel learns its own projection layers. The projections use multichannel linear layers with no depth-wise transform and then uses separable 1x9 2d convolutions in the same way the primer architecture uses 1d convolutions. This makes use of Query-Key normalization as it leads to more diffuse attention which is ideal for spectrograms; this normalization is applied to each heads features individually. This attention mechanism also makes use of rotary positional embeddings.

* **FrameTransformerEncoder** - This is the transformer encoder module. It is a pre-norm variant of the transformer encoder architecture which makes use of multichannel multihead attention and multichannel linear layers to allow for parallel transformers effectively; aside from that it is the same as typical transformers. As in the primer architecture, this makes use of squared relu for activation.

* **FrameTransformerDecoder** - This is the transformer decoder module. It is a pre-norm variant of the transformer decoder architecture which makes use of multichannel multihead attention and multichannel linear layers to allow for parallel transformers effectively; aside from that it is the same as typical transformers. For memory, this makes use of the position-wise residual u-nets skip connection. As in the primer architecture, this makes use of squared relu for activation.

## References
- [1] Jansson et al., "Singing Voice Separation with Deep U-Net Convolutional Networks", https://ismir2017.smcnus.org/wp-content/uploads/2017/10/171_Paper.pdf
- [2] Takahashi et al., "Multi-scale Multi-band DenseNets for Audio Source Separation", https://arxiv.org/pdf/1706.09588.pdf
- [3] Takahashi et al., "MMDENSELSTM: AN EFFICIENT COMBINATION OF CONVOLUTIONAL AND RECURRENT NEURAL NETWORKS FOR AUDIO SOURCE SEPARATION", https://arxiv.org/pdf/1805.02410.pdf
- [4] Liutkus et al., "The 2016 Signal Separation Evaluation Campaign", Latent Variable Analysis and Signal Separation - 12th International Conference
- [5] Vaswani et al., "Attention Is All You Need", https://arxiv.org/pdf/1706.03762.pdf
- [6] So et al., "Primer: Searching for Efficient Transformers for Language Modeling", https://arxiv.org/pdf/2109.08668v2.pdf
- [7] Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding", https://arxiv.org/abs/2104.09864
- [8] Henry et al., "Query-Key Normalization for Transformers", https://arxiv.org/pdf/2010.04245.pdf
- [9] Asiedu et all., "Decoder Denoising Pretraining for Semantic Segmentation", https://arxiv.org/abs/2205.11423
