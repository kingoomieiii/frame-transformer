FROM nvidia/cuda:11.3.1-cudnn8-runtime

RUN apt-get -y update \
    && apt-get install -y --no-install-recommends \
        wget \
        curl \
        python3 \
        python3-pip \
        libsndfile1

RUN pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
RUN pip3 install --upgrade pip
RUN pip3 install tqdm opencv_python

WORKDIR /root
COPY app /root/app