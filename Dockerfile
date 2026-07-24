FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y python3.10 python3-pip git libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

RUN pip3 install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118 

RUN pip3 install "numpy<2" matplotlib opencv-python imageio scikit-image tqdm "cupy-cuda11x<13"

WORKDIR /workspace