# vocal-remover

[![Release](https://img.shields.io/github/release/tsurumeso/vocal-remover.svg)](https://github.com/tsurumeso/vocal-remover/releases/latest)
[![Release](https://img.shields.io/github/downloads/tsurumeso/vocal-remover/total.svg)](https://github.com/tsurumeso/vocal-remover/releases)

This is a deep-learning-based tool to extract instrumental track from your songs.

### From Ben Carper
This is a variation of tsurumeso's vocal remover that I've been tinkering with for a while that I'm calling a frame transformer. The goal of this project was to find a meaningful way to use the evolved transformer architecture for track separation. This isn't a very user-friendly version just yet, has a lot of stuff that is specific to my dataset and environment; will take at least 12gb of VRAM to train this given the settings in the code. Will pretty this up over time.

This version consists of only a single u-net. This u-net includes modified evolved transformer encoders after each downsampling as well as modified evolved transformer decoders before each upsampling.

Will update this repo in the coming days with checkpoints with listed settings. My personal dataset consists of 5,061 instrumental songs, 488 vocal tracks, and 274 instrumental + mix song pairs.

The key difference between this architecture and the original is the use of a modified evolved transformer architecture as well as all 2d convolutions use Nx1 kernels. This is to avoid transferring information between frames before desired; in the code the Nx1 convs are called frame convolutions for lack of a better name (perhaps this has a better name already, no idea). Only features from the same frame are encoded with each other and all information shared between frames occurs only at the wide 1d convolutions or the multiband frame attention in the transformer modules. After each downsampling a sequence of transformer encoders is ran which are connected in a sort of DenseNet setup, where each layer feeds into the subsequent layer along with the original input. I did this due to the bottlenecking that is used for using the transformer architecture with the convolutional encoder/decoder thinking that it would help integration of information from the various channels. One alternative is to include the previous encoders output and simply add the bottleneck to it with layer norm and activation as with other residual connections in the transformer architecture, but quick tests didn't show a significant change. Modified evolved transformer decoders are used after each up sampling except the last, using the skip connection as memory and having a separate bottleneck learned for it (not sure if this helps; also need to test adding in the transformer encoder/decoder for the full resolution to see if it helps). Testing changes to the use of transformer modules at every resolution; current version uses a transformer encoder and decoder at every resolution, the last decoder being 2 transformer decoders that are concatenated and sent through a sigmoid to produce the final output.

This fork also makes use of vocal augmentations. This takes a random vocal spectrogram and adds it to an isntrumental spectrogram for training; the dataset takes a path to a directory with vocal spectrogram npz files (need to update the dataset generator code still, the new vocal augmentation dataset expects unnormalied magnitudes with the normalization coefficient set in its own property in the npz file; old dataset class is still in dataset py).

A test has been carried out between the original stacked u-net implementation that was modified to use frame convolution encoders and this to verify that the boost in performance was not from the frame convolutions and rather from the transformer architecture. The 3x1 encoders used along with the vanilla architecture causes training time to literally double from 4 hours to 8 hours on my dataset while also causing the loss to be significantly higher. I need to do a test between the original architecture with its 3x3 encoders and this, however vocal extractions seem to be subjectively higher quality.

## Installation

### Getting vocal-remover
Download the latest version from [here](https://github.com/tsurumeso/vocal-remover/releases).

### Install PyTorch
**See**: [GET STARTED](https://pytorch.org/get-started/locally/)

### Install the other packages
```
cd vocal-remover
pip install -r requirements.txt
```

## Usage
The following command separates the input into instrumental and vocal tracks. They are saved as `*_Instruments.wav` and `*_Vocals.wav`.

### Run on CPU
```
python inference.py --input path/to/an/audio/file
```

### Run on GPU
```
python inference.py --input path/to/an/audio/file --gpu 0
```

### Advanced options
`--tta` option performs Test-Time-Augmentation to improve the separation quality.
```
python inference.py --input path/to/an/audio/file --tta --gpu 0
```

`--postprocess` option masks instrumental part based on the vocals volume to improve the separation quality.  
**Experimental Warning**: If you get any problems with this option, please disable it.
```
python inference.py --input path/to/an/audio/file --postprocess --gpu 0
```

## Train your own model

### Place your dataset
```
path/to/dataset/
  +- instruments/
  |    +- 01_foo_inst.wav
  |    +- 02_bar_inst.mp3
  |    +- ...
  +- mixtures/
       +- 01_foo_mix.wav
       +- 02_bar_mix.mp3
       +- ...
```

### Train a model
```
python train.py --dataset path/to/dataset --reduction_rate 0.5 --mixup_rate 0.5 --gpu 0
```

## References
- [1] Jansson et al., "Singing Voice Separation with Deep U-Net Convolutional Networks", https://ismir2017.smcnus.org/wp-content/uploads/2017/10/171_Paper.pdf
- [2] Takahashi et al., "Multi-scale Multi-band DenseNets for Audio Source Separation", https://arxiv.org/pdf/1706.09588.pdf
- [3] Takahashi et al., "MMDENSELSTM: AN EFFICIENT COMBINATION OF CONVOLUTIONAL AND RECURRENT NEURAL NETWORKS FOR AUDIO SOURCE SEPARATION", https://arxiv.org/pdf/1805.02410.pdf
- [4] Liutkus et al., "The 2016 Signal Separation Evaluation Campaign", Latent Variable Analysis and Signal Separation - 12th International Conference
- [5] So, Liang, and Le, "The Evolved Transformer", https://arxiv.org/pdf/1901.11117.pdf
