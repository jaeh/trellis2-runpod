# TRELLIS.2 RunPod Serverless - Best Practices Version
# - Minimal image (no model weights baked in)
# - Uses RunPod's built-in model caching
# - linux/amd64 platform required

FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

ENV PYTHONUNBUFFERED=1
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Use RunPod's cached model path
ENV HF_HOME=/runpod-volume/huggingface-cache

WORKDIR /app

# Install system dependencies (minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    ninja-build \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Remove conflicting distutils packages from base image
RUN rm -rf /usr/lib/python3/dist-packages/blinker* || true

# Install Python dependencies
RUN pip install --no-cache-dir \
    runpod \
    huggingface_hub \
    accelerate \
    safetensors \
    imageio \
    imageio-ffmpeg \
    opencv-python-headless \
    trimesh \
    easydict \
    einops \
    kornia \
    timm \
    omegaconf \
    xformers \
    pymeshlab \
    open3d \
    plyfile \
    git+https://github.com/EasternJournalist/utils3d.git

# Clone TRELLIS.2 and install
RUN git clone --recursive https://github.com/microsoft/TRELLIS.2.git /app/TRELLIS.2 \
    && cd /app/TRELLIS.2 \
    && pip install --no-cache-dir -r requirements.txt || true \
    && pip install --no-cache-dir git+https://github.com/NVlabs/nvdiffrast.git || true \
    && pip install --no-cache-dir -e . || true

ENV TRELLIS_PATH=/app/TRELLIS.2

# Copy handler (small file)
COPY handler.py /app/handler.py

# NO model download here - RunPod's model caching handles this
# Model will be at: /runpod-volume/huggingface-cache/hub/models--microsoft--TRELLIS.2-4B/

CMD ["python3", "-u", "handler.py"]
