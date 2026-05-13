# Uporabimo uradno PyTorch sliko s CUDA podporo
FROM pytorch/pytorch:1.7.1-cuda11.0-cudnn8-runtime

# Sistemski paketi, ki jih OpenCV pogosto potrebuje
RUN apt-get update && apt-get install -y \
    git \
    vim \
    curl \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Python paketi
RUN pip install --upgrade pip
RUN pip install numpy
RUN pip install nibabel
RUN pip install timm==0.5.4
RUN pip install scipy
RUN pip install matplotlib
RUN pip install openpyxl
RUN pip install opencv-python
RUN pip install pandas
RUN pip install scikit-image
RUN pip install scikit-learn
RUN pip install jupyter
RUN pip install ipykernel
RUN pip install tqdm
RUN pip install mediapipe

WORKDIR /workspace
