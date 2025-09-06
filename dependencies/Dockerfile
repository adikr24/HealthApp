# Specify the Dockerfile syntax to ensure latest buildkit features are used.
# syntax=docker/dockerfile:1.4
#### worked on detectron2 and slowfast remove MMACTION 2 if build fails ###
# Use the CUDA 12.8 devel base image with Ubuntu 22.04.
# This ensures compatibility with your host's CUDA 12.8 installation and includes nvcc.
FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

# Set environment variables for non-interactive installation.
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHON_VERSION=3.9
# You can update this if you target a specific Detectron2 release
ENV DETECTRON2_VERSION=0.6

# Install minimal base dependencies, including software-properties-common for adding PPAs,
# build tools (ninja-build, git), network tools (wget, curl, ca-certificates),
# graphical display libraries, and video processing (ffmpeg).
# python3-pip is included here as it's part of the base apt repos.
# Python3.9 and its dev components, and cuDNN will be installed AFTER adding the PPA.
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    ca-certificates \
    git \
    wget \
    sudo \
    ninja-build \
    libgl1-mesa-glx \
    libsm6 \
    libxext6 \
    ffmpeg \
    python3-pip && \
    rm -rf /var/lib/apt/lists/*

# Add the deadsnakes PPA to install Python 3.9 and its specific packages.
# This MUST happen BEFORE attempting to install python3.9 packages.
RUN add-apt-repository ppa:deadsnakes/ppa && apt-get update

# Now install python3.9 and its dependencies (including cuDNN).
# These packages are now discoverable by apt after the PPA was added and updated.
RUN apt-get install -y --no-install-recommends \
    python3.9 \
    python3.9-dev \
    libcudnn8 \
    libcudnn8-dev && \
    rm -rf /var/lib/apt/lists/*

# Install the distutils module for python3.9, required by pip on some systems.
# An apt-get update is included here to ensure distutils is found after previous installations.
RUN apt-get update && apt-get install -y python3.9-distutils

# Set Python 3.9 as the default Python and pip within the container.
# This ensures all subsequent python/pip commands use the correct version.
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.9 1
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1
RUN update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Create a non-root user for security best practices.
ARG USER_ID=1000
RUN useradd -m --no-log-init --system --uid ${USER_ID} appuser -g sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER appuser
WORKDIR /home/appuser

# Add user's local bin directory to PATH to pick up user-installed packages.
ENV PATH="/home/appuser/.local/bin:${PATH}"

# Set CUDA_HOME and add CUDA bin and lib64 directories to PATH and LD_LIBRARY_PATH.
# This ensures PyTorch and Detectron2 compilation can find CUDA tools like nvcc and CUDA libraries.
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Install PyTorch nightly with CUDA 12.8 support.
# This uses the specific nightly wheel index URL for CUDA 12.8.
RUN pip install --user --no-cache-dir --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

# Install general dependencies for Detectron2 and SlowFast, including opencv-python-headless.
# This includes items from the requirements lists like simplejson, psutil, tensorboard, moviepy, PyAV, PyTorchVideo, pycocotools.
# Note: cmake, fvcore, numpy, iopath, omegaconf, hydra-core, packaging are already installed or will be.
RUN pip install --user --no-cache-dir \
    simplejson \
    psutil \
    tensorboard \
    moviepy \
    av \
    pytorchvideo \
    'git+https://github.com/facebookresearch/fvcore' \
    'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI' \
    cmake \
    numpy==1.24.4 \
    opencv-python-headless \
	Pillow

# Clone Detectron2 repository.
RUN git clone https://github.com/facebookresearch/detectron2 detectron2_repo
WORKDIR /home/appuser/detectron2_repo

RUN pip install --user --no-cache-dir Pillow

# Set Detectron2-specific environment variables for building.
ENV FORCE_CUDA="1"
ARG TORCH_CUDA_ARCH_LIST="8.6;9.0" # Adjust if your GPU's compute capability is different
ENV TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}"

# Install Detectron2 from source in editable mode.
RUN pip install --user --no-cache-dir -e .

# Install SlowFast
WORKDIR /home/appuser
RUN git clone -b fix/pillow-dependency https://github.com/adikr24/SlowFast.git pyscene-src

ENV PYTHONPATH="/home/appuser/pyscene-src"
WORKDIR /home/appuser/pyscene-src 

# --- DEBUGGING STEPS START ---
RUN echo "--- Original setup.py content ---" && cat setup.py
# Optional fix if 'PIL' ever appears again
RUN sed -i "s/'PIL'/'Pillow'/g" setup.py
# --- DEBUGGING STEPS END ---
RUN echo "--- Modified setup.py content ---" && cat setup.py

# Prevent sklearn error
ENV SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=true

# Install SlowFast and dependencies
RUN pip install --user --no-cache-dir -e .

# Install Yolo
RUN pip install --user --no-cache-dir ultralytics
# Install Pillow-heif (for image conversion from HEIC to .jpg)
RUN pip install --user --no-cache-dir pillow-heif

# Set a fixed model cache directory.
ENV FVCORE_CACHE="/tmp"


# MMAction2 (prebuilt mmcv for CUDA 12.8 + torch 2.5 nightly)
RUN pip install --user --no-cache-dir openmim && \
    PATH=$PATH:/home/appuser/.local/bin && \
    pip install --user --no-cache-dir \
        mmcv==2.1.0 \
        -f https://download.openmmlab.com/mmcv/dist/cu128/torch2.5/index.html && \
    pip install --user --no-cache-dir \
        git+https://github.com/open-mmlab/mmaction2.git

ENV PYTHONPATH="/home/appuser/.local/lib/python3.9/site-packages:${PYTHONPATH}"


# Launch into bash
CMD ["bash"]
