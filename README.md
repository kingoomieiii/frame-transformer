# FrameTransformer / AsymUNetMCT

This fork is beginning to grow out of the research phase, changes will likely start slowing down. I have one more step I would like to take as far as a new type of GAN, but for now my focus will be on finishing the new frame transformer's training; experiments are showing very strong results with the newer version. The new architecture (and really V1 as well) I am now calling an AsymUNetMCT, or **Asym**metrically down-sampled **UNet** **M**ulti-**C**hannel **T**ransformer. I have started working in a new folder and will finalize the older frame transformer models along with checkpoints so people can make use of those (haven't tried using them as an ensemble but that would probably work well honestly, they do have slightly different behavior).

I'm shifting work into the app folder for docker related reasons, other folders will no longer be updated and will eventually be deleted. The app folder contains scripts that are able to run using DistributedDataParallel, the dockerfile for this is located in the root directory with a build.sh script to build it for convenience. You can run these directly without docker, however.

Asymmetrically down-sampled U-Net refers to a U-Net that down-samples with a stride of (2,1) rather than the conventional stride of 2. Multichannel transformers are an extension of transformers that I created for use with multichannel audio data.

ConvolutionalEmbedding - This module was inspired by [11] below. This makes use of an encoding branch that consists of residual blocks with symmetrical stride, as well as residual blocks for extracting positional information from that scale. The down-sampling blocks utilize a kernel size of 3 while the position extraction layers utilize kernel sizes of 11, as more padding is shown to help with positional information in the paper. This also includes sinusoidal positional encoding in the form of an extra channel that is appended to the input before being down-sampled.

FrameConv - This module is focused on being a layer that utilizes locality information along with the fully connected linear layers in order to contextualize the fully connected features. It makes use of a symmetrical kernel convolution followed by batched-matrix multiplication to carry out parallel linear layers on the GPU in an efficient manner.

MultichannelLinear - This module is focused on being a layer that processes parallel linear layers and then shares information between those layers. It makes use of batched-matrix multiplication to process parallel linear layers on the GPU and then utilizes matrix multiplication again to carry out a depth-wise transformation that is equivalent to a 1x1 convolution in order to share information between frames and to modify number of channels.

MultichannelLayerNorm - This module applies LayerNorm in parallel across multiple channels, allowing each channel to learn its own element-wise affine parameters.

ChannelNorm - This module applies LayerNorm across the channel dimension, module is included for ease of use.

ResBlock - This is a res block that is more inspired by the pre-norm transformer architecture than resnet. It uses less computational resources while performing similarly which allowed me to scale the transformer portion of the network a bit more. This utilizes MultichannelLayerNorm for normalization.

FrameEncoder - This is the encoder for the asymmetrical u-net; it consists of a single residual block with a symmetrical kernel but asymmetrical stride when downsampling so as to preserve the temporal dimension.

FrameDecoder - This is the decoder for the asymmetrical u-net; it consists of a single residual block that operates on the bilinearly interpolated input with the skip connection concatenated to the channel dimension.

ConvolutionalMultiheadAttention - This module is a variant of multihead attention that I created to use within the latent space of the U-Net. This is similar to other approaches that are in use elsewhere, I opted to include a symmetrical kernel size > 1 and use multi-head attention. This includes an optional residual attention connection as well.

MultichannelMultiheadAttention - This module was created to utilize parallel attention mechanisms. It uses rotary positional embedding as seen in the RoFormer architecture to better contextualize positional information. After this, the query, key, and value projections utilize MultichannelLinear layers with no depth-wise component which is then followed by 1xN kernel convolutions that utilize only a single group so as to share information across the attention mechanisms. This includes a residual attention connection and is ended with a final output projection that utilizes a multichannel linear layer.

FrameTransformerEncoder - Frame transformer encoders work by optionally extracting a set of feature maps from the overall channel volume in a similar manner to MMDENSELSTM when it extracts a single channel and then computing attention between the frames of the extracted channels in parallel. The architecture for this transformer encoder is modeled off of a few transformer variants. The main structure is taken from the Evolved Transformer and then extended into the channel dimension using the above multichannel layers and FrameConv, while also making use of changes from the Primer architecture, the RealFormer architecture, and the RoFormer architecture. Where the Evolved Transformer utilizes depth-wise separable 1d convolutions, this architecture utilizes the above FrameConv modules. This is similar to how the separable convolutions work in the original Evolved Transformer paper, however instead of the frequency dimension being fully connected it too uses locality followed by the fully connected parallel linear layers. This concatenates the output to the input, returns the output separately for use later in the network, and returns a residual attention connection for use in the next layer as well as later on in the network.

FrameTransformerDecoder - Frame transformer decoders work by optionally extracting a set of feature maps from the overall channel volume in a similar manner to MMDENSELSTM when it extracts a single channel and then computing attention between the frames of the extracted channels in parallel. As above, this is based on a series of transformer architectures that have been adapted for multichannel audio data. The main difference between this and the encoder is that this follows the Evolved Transformer Decoder architecture. For memory, this module utilizes the returned attention maps from the corresponding FrameTransformerEncoder from that level. For the residual attention connection for skip attention, there are two residual connections. To deal with this, I introduced a gating mechanism which uses 3d convolutions to process the attention scores from the residual attention connections and outputs a weight for each frequency that allows the network to choose which source to use information from.

ConvolutionalTransformerEncoder - This is a transformer encoder that is uses the channel dimension as the embedding dimensions. This too utilizes the evolved transformer architecture, however it has been adapted for use with multichannel data. This utilizes ChannelNorm to normalize features across the channel dimension so as to contextualize the channels of each feature and allow to compare them with one another in a more sensible manner. This makes use of ConvolutionalMultiheadAttention.

V1 - This neural network has been trained to about 1.2 million optimization steps using progressively larger context lengths throughout training. It makes use of rotary positional embeddings only, and consists of only a single attention map per u-net layer. It uses far less vram than the other variants, however. I need to update the script to play nicely with this checkpoint, will fix that shortly but shouldn't be too hard for people if they are familiar with PyTorch. https://mega.nz/file/C5pGXYYR#ndHuj-tYWtttoj8Y4QqAkAruvZDYcQONkTHfZoOyFaQ

V2b - This neural network is similar to the new frame transformer, however it uses the conventional transformer architecture and does not include a convolutional transformer branch. https://mega.nz/file/a843wbTQ#E319mlp5qjsyky6PRp-zikou-YtWb9TVpbtHGaVh2AA

V3 - Need to get this architectures checkpoint from my other machine, will update afterward. This is a frame transformer that doesn't downsample at all.

## References
- [1] Jansson et al., "Singing Voice Separation with Deep U-Net Convolutional Networks", https://ismir2017.smcnus.org/wp-content/uploads/2017/10/171_Paper.pdf
- [2] Takahashi et al., "Multi-scale Multi-band DenseNets for Audio Source Separation", https://arxiv.org/pdf/1706.09588.pdf
- [3] Takahashi et al., "MMDENSELSTM: AN EFFICIENT COMBINATION OF CONVOLUTIONAL AND RECURRENT NEURAL NETWORKS FOR AUDIO SOURCE SEPARATION", https://arxiv.org/pdf/1805.02410.pdf
- [4] Liutkus et al., "The 2016 Signal Separation Evaluation Campaign", Latent Variable Analysis and Signal Separation - 12th International Conference
- [5] Vaswani et al., "Attention Is All You Need", https://arxiv.org/pdf/1706.03762.pdf
- [6] So et al., "Primer: Searching for Efficient Transformers for Language Modeling", https://arxiv.org/pdf/2109.08668v2.pdf
- [7] Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding", https://arxiv.org/abs/2104.09864
- [9] Asiedu et all., "Decoder Denoising Pretraining for Semantic Segmentation", https://arxiv.org/abs/2205.11423
- [10] He et al., "RealFormer: Transformer Likes Residual Attention", https://arxiv.org/abs/2012.11747
- [11] Islam et al., "How Much Position Information Do Convolutional Neural Networks Encode?", https://arxiv.org/abs/2001.08248
- [12] Isola et al., "Image-to-Image Translation with Conditional Adversarial Networks", https://arxiv.org/pdf/1611.07004.pdf
