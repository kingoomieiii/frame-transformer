# frame-transformer

This is a collection of tools for working with spectrograms using transformers. It includes a vocal remover that has evolved from tsurumeso's vocal remover, a phase predictor, pretraining, a gan, and a vector quantizer for frames. Right now I am working on training the vocal remover, however I will be moving onto these others after that is complete. GAN script isn't ready (mnight work but I haven't really tried). My plan is to finish up training of the current vocal remover and then work on a conditional gan using the frame transformer architecture similar to how pix2pix works.

I'm calling this architecture a frame transformer. The neural network is a position-wise linear residual u-net coupled with what I call a multichannel transformer. Current training is going very well at 62459 optimization steps on a cropsize of 256, my plan is to allow it to run to 150k optimization steps at a cropsize of 256 and then up it to 512 to allow for it to build more of an understanding of long distance relationships between frames of audio. For this model I will likely only take it to 250k optimization steps. Given the method of training, to an extent this is self-supervised in that it constructs training data from instrumental tracks and vocal tracks rather than using a fixed dataset. I suspect once trained the model will be able to be finetuned for downstream tasks such as remastering. Edit: Validation loss is still steadily dropping after 48 hours of training... Think this is going to take a while to unlock its full potential lol.

This neural network at its core relies on a type of layer that I refer to as a multichannel linear layer. This has two weight tensors: a 3d weight tensor which is the weight matrices for each channels position-wise transform and then a 2d weight matrix for the depth-wise transform. This allows each channel to have its own position-wise linear layer that is applied to each frame while taking advantage of batched matrix multiplication. Compared to conv1d, this is around 2x faster when using smaller numbers of channels and far faster when using many channels/parallel linear layers.

Technically, this transformer can be applied to typical problems that transformers deal with as well. You could embed a sequence of shape [B,W] for a shape of [B,W,H] and then transpose dimensions 1 and 2 and unsqueeze at dimension 1. From here, the frame transformer can expand the single channel embeddings into multiple channels which could be viewed as different representations of the embedding vector. Not sure if this would help in the world of natural language processing, but it clearly helps with audio so who knows.

This fork also makes use of a dataset I refer to as voxaug in order to satisfy the transformers need for large amounts of data. This dataset randomly selects from a library of instrumental music and a library of vocal tracks and mixes them together for the neural network to train on. This has the benefit of inflating data exponentially as well as ensuring data is perfect for the removal process. To an extent you could view this as self-supervised learning in that its learning to remove a mask of vocals. My instrumental dataset consists of 30.88 days worth of music while my vocal stem library consists of 1416 full song vocal tracks. I will be uploading checkpoints for a 357,493,618 parameter model after it trains for a few days.

Current training at a cropsize of 256; orange at the bottom is on 10 seconds of audio, others are at 5 seconds. Will be increasing to 10 seconds with the current version after I reach 150k steps which should see it overtake the orange which is my best run yet with the convolutional varant. Green is a run with the convolutional variant at a cropsize of 256. Comparing with the parent repo, at around 62459k steps it was competitive with the original with a more full mix, so it should be interesting to see where this is now at 108k steps and eventually once I get to 250k ![image](https://user-images.githubusercontent.com/30326384/188479869-a7608716-4038-4afe-8c90-9c983a6e9ee4.png)

I might let this train for longer, however I have the max step count set to 350k which is 150k less than the primer architecture and 650k less than BERT. I suspect once trained to matching number of steps this architecture will be able to be finetuned fairly well for downstream tasks such as remastering.



## Architecture Diagram ##
![image](https://user-images.githubusercontent.com/30326384/188337890-d92faa0e-0565-4a2a-9e2e-e678f11b4f6b.png)

## Module Descriptions ##

* **FrameTransformer** - The core of the neural network. This consists of a series of encoders and decoders, with encoders defined as frame_transformer_encoder(frame_encoder(x)) and decoders defined as frame_transformer_decoder(frame_decoder(x, skip), skip). It also includes an output depthwise linear layer in the form of a weight matrix.

* **MultichannelLinear** - This was my solution to having parallel linear layers. Instead of having individual linear layers, I compressed them into a single weight matrix with a channel dimension and make use of batched matrix multiplication. It also includes a depth-wise linear layer for increasing channel count (compression of frequency axis and expansion of channels is still necessary for this to learn well, although it seems to have less of a reliance on channels than a convolutional neural network).

* **FrameNorm** - This applies layer norm to each frame; each channel has its own element-wise affine parameters.

* **FrameEncoder** - position-wise encoder for each frame responsible for downsampling and expansion of channels. This consists of a residual block made from multichannel linear layers to allow for each channel to learn its own position-wise linear layer. It takes inspiration from the transformer architecture and uses residual blocks in that style - linear2(activation(linear1(norm(x)))). For activation this uses squared ReLU as in the primer paper.

* **FrameDecoder** - position-wise decoder for each frame responsible for upsampling and reduction of channels. This consists of two residual blocks; the first allows each channel to learn its own position-wise and depth-wise residual block for upsampling frames, and the second residual block integrates the skip connection by concatenating it with the output of the first block and reducing it back to out_channels with a position-wise and depth-wise multichannel linear layer. For activation this uses squared ReLU as in the primer paper.

* **MultichannelMultiheadAttention** - This module is an extension of multihead attention for multiple channels. Each channel learns its own projection layers. The projections use multichannel linear layers; this no longe rmakes use of convolutions as in the primer architecture, though once I train this version out I'll try a variant with 1x3 kernel convolutions (1x9 completely broke training so yeah, not fully convinced convolutions are useful for this problem).

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
