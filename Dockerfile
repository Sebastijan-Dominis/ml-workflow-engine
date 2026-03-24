# conda-lock causes major issues, so it is avoided here.
# Note that a docker build sometimes creates the environment from scratch,
# even when it shouldn't, which is not ideal for development speed. 
# However, it ensures that the environment is always clean and consistent with the environment.yml file. 
# For faster development iterations, consider using a local Conda environment on your machine with the 
# same environment.yml file.

# ===== Base image with GPU support =====
FROM pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime
# NOTE: Uncomment the line below and comment out the line above only if you want to generate fake data.
# For regular use, stick to the current image. If you use the image below, make sure to also uncomment
# the torch installation line in this Dockerfile. Likewise, uncomment the sdv dependency in 
# requirements.txt.

# FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

# ===== Set working directory =====
WORKDIR /app

# ===== Timezone fix =====
ENV TZ=Europe/Zagreb
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# ===== Install Miniconda and Git =====
RUN apt-get update && apt-get install -y wget bzip2 git && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh && \
    /opt/conda/bin/conda clean -afy

# Add conda to PATH
ENV PATH="/opt/conda/bin:$PATH"

# ===== Accept Conda Terms of Service =====
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# ===== Copy environment.yml first for faster rebuilds =====
COPY environment.yml /tmp/environment.yml

# ===== Create Conda environment =====
# Create env
RUN conda env create -f /tmp/environment.yml -n hotel_management

# Copy the rest of the files needed for installation
COPY requirements.txt /tmp/requirements.txt
COPY setup.py .
COPY pyproject.toml .

# Install torch (nightly CUDA 12.8)
# This version works with Nvidia RTX 5070 Ti GPU. If you experience issues, change to a compatible version for your GPU.

# NOTE: Uncomment this only if you want to generate fake data. In that case, use the commented out base
# image at the top of this file. Likewise, uncomment the sdv dependency in requirements.txt. For regular 
# use, stick to the current image.

# RUN conda run -n hotel_management pip install --pre \
#     torch==2.12.0.dev20260320+cu128 \
#     --index-url https://download.pytorch.org/whl/nightly/cu128

# Install rest
RUN conda run -n hotel_management pip install -r /tmp/requirements.txt

# ===== Use the environment for all container commands =====
SHELL ["conda", "run", "-n", "hotel_management", "/bin/bash", "-c"]

# ===== Copy the code =====
COPY ml ./ml
COPY pipelines ./pipelines
COPY scripts ./scripts
COPY ml_service ./ml_service

# Install the package
RUN conda run -n hotel_management pip install -e .

# ===== Expose ports =====
EXPOSE 8000
EXPOSE 8050

# ===== Run all services =====
CMD bash -c "\
    uvicorn ml_service.backend.main:app --reload --host 0.0.0.0 --port 8000 & \
    python -m ml_service.frontend.app --host 0.0.0.0 --port 8050 & \
    wait"