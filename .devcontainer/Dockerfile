FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu24.04

LABEL maintainer="Data Science Project"
LABEL description="Dev container TensorFlow GPU - CUDA 12.8 - RTX 5070"

# Variables d'environnement
ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    CONDA_DIR=/opt/conda \
    PATH=/opt/conda/bin:$PATH

# Installation des dépendances système (Ubuntu 24.04 compatible)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    wget \
    git \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Installation de Miniforge (Conda + Mamba)
RUN curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh" \
    && bash Miniforge3-Linux-x86_64.sh -b -p $CONDA_DIR \
    && rm Miniforge3-Linux-x86_64.sh \
    && $CONDA_DIR/bin/conda clean -afy

WORKDIR /workspace

# Copie de environment.yml
COPY environment.yml /tmp/environment.yml

# Création de l'environnement gpu-env
RUN $CONDA_DIR/bin/mamba env create -f /tmp/environment.yml \
    && $CONDA_DIR/bin/conda clean -afy \
    && rm /tmp/environment.yml

# Activation automatique de l'environnement
RUN echo "source $CONDA_DIR/etc/profile.d/conda.sh" >> /root/.bashrc \
    && echo "conda activate gpu-env" >> /root/.bashrc

# Variables pour l'environnement actif
ENV CONDA_DEFAULT_ENV=gpu-env \
    PATH=/opt/conda/envs/gpu-env/bin:$PATH

EXPOSE 8888 6006

CMD ["/bin/bash"]
