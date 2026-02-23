# TRELLIS.2 RunPod Serverless
# - Model baked into container at build time (~12GB image)

FROM runpod/pytorch:2.6.0-py3.11-cuda12.6.1-cudnn-devel-ubuntu22.04

ENV PYTHONUNBUFFERED=1
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ENV OPENCV_IO_ENABLE_OPENEXR=1
ENV HF_HOME=/runpod-volume/huggingface-cache
ENV CUDA_HOME=/usr/local/cuda
ENV PATH="${CUDA_HOME}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"
# Set CUDA architectures (9.0=RTX 5090, 9.0=H100, 8.0=A100)
ENV TORCH_CUDA_ARCH_LIST="8.0;9.0"

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libjpeg-dev \
    ninja-build \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Remove conflicting distutils packages
RUN rm -rf /usr/lib/python3/dist-packages/blinker* || true

# Install build tools first (required for nvdiffrast)
RUN pip install --no-cache-dir --upgrade pip setuptools wheel ninja

# Install torch and torchvision (CUDA 12.6 compatible versions)
RUN pip install --no-cache-dir torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu126
RUN pip install --no-cache-dir \
    runpod \
    huggingface_hub \
    accelerate \
    safetensors \
    imageio \
    imageio-ffmpeg \
    pillow \
    opencv-python-headless \
    trimesh \
    easydict \
    einops \
    kornia \
    timm \
    omegaconf \
    tqdm \
    lpips \
    transformers \
    plyfile \
    pymeshlab \
    open3d \
    git+https://github.com/EasternJournalist/utils3d.git

# Install xformers with correct CUDA 12.6 index
RUN pip install --no-cache-dir xformers --index-url https://download.pytorch.org/whl/cu126

# Install nvdiffrast v0.4.0 (CUDA extension - requires setuptools, wheel, ninja)
RUN git clone -b v0.4.0 https://github.com/NVlabs/nvdiffrast.git /tmp/nvdiffrast \
    && pip install --no-cache-dir --no-build-isolation /tmp/nvdiffrast \
    && rm -rf /tmp/nvdiffrast

# Install nvdiffrec (renderutils) - using JeffreyXiang's fork as required by TRELLIS.2
RUN git clone -b renderutils https://github.com/JeffreyXiang/nvdiffrec.git /tmp/nvdiffrec \
    && pip install --no-cache-dir --no-build-isolation /tmp/nvdiffrec \
    && rm -rf /tmp/nvdiffrec

# Install cumesh (CUDA mesh operations)
RUN git clone https://github.com/JeffreyXiang/CuMesh.git /tmp/CuMesh --recursive \
    && pip install --no-cache-dir --no-build-isolation /tmp/CuMesh \
    && rm -rf /tmp/CuMesh

# Install FlexGEMM (CUDA GEMM operations)
RUN git clone https://github.com/JeffreyXiang/FlexGEMM.git /tmp/FlexGEMM --recursive \
    && pip install --no-cache-dir --no-build-isolation /tmp/FlexGEMM \
    && rm -rf /tmp/FlexGEMM

# Install flash-attn from pre-built wheel (v2.7.4 for CUDA 12.6 + PyTorch 2.6)
RUN pip install --no-cache-dir \
    https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.0.8/flash_attn-2.7.4+cu126torch2.6-cp311-cp311-linux_x86_64.whl

# Clone TRELLIS.2
RUN git clone --recursive https://github.com/microsoft/TRELLIS.2.git /app/TRELLIS.2

# Install o-voxel from TRELLIS.2
RUN pip install --no-cache-dir --no-build-isolation /app/TRELLIS.2/o-voxel

# Pre-download TRELLIS.2-4B model to container
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('microsoft/TRELLIS.2-4B', local_dir='/app/TRELLIS.2-4B')"

# Add TRELLIS.2 to Python path (it's not a pip package)
ENV PYTHONPATH="/app/TRELLIS.2"
ENV TRELLIS_PATH=/app/TRELLIS.2

# Copy handler
COPY handler.py /app/handler.py

CMD ["python3", "-u", "handler.py"]