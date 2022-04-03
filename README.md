# vocal-remover

[![Release](https://img.shields.io/github/release/tsurumeso/vocal-remover.svg)](https://github.com/tsurumeso/vocal-remover/releases/latest)
[![Release](https://img.shields.io/github/downloads/tsurumeso/vocal-remover/total.svg)](https://github.com/tsurumeso/vocal-remover/releases)

This is a deep-learning-based tool to extract instrumental track from your songs.

### From Ben Carper
This is a variation of tsurumeso's vocal remover that I've been tinkering with for a while. The goal of this project was to find a meaningful way to use the transformer architecture for track separation.

This version consists of only a single u-net. This u-net includes modified evolved transformer encoders after each downsampling as well as modified evolved transformer decoders before each upsampling.

The key difference with this architecture is that all encoders use 3x1 convolutions and when downsampling use a stride of (2,1), which I'll call column convolutions for lack of a better name (perhaps this concept exists, no idea). This way, only features from the same frame are encoded with each other and all information shared between frames occurs only at the wide 1d convolutions or the multihead frame attention in the transformer modules. After each downsampling a sequence of transformer encoders is ran which are connected in a sort of DenseNet setup, where each layer feeds into the subsequent layer along with the original input. I did this due to the bottlenecking that is used for using the transformer architecture with the convolutional encoder/decoder. One alternative is to include the previous encoders output and simply add the bottleneck to it with layer norm and activation as with other residual connections in the transformer architecture, but quick tests didn't show a significant change.

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
