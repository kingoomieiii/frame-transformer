FROM nvidia/cuda:12.0.1-cudnn8-runtime-ubuntu22.04

RUN apt-get -y update \
    && apt-get install -y --no-install-recommends \
        wget \
        curl \
        iputils-ping

RUN apt-get install -y --no-install-recommends python3 python3-pip libsndfile1

        # python3 \
        # python3-pip \
        # libsndfile1

RUN pip3 install torch torchvision torchaudio
RUN pip3 install --upgrade pip
RUN pip3 install tqdm opencv_python einops librosa

WORKDIR /root
COPY app /root/app