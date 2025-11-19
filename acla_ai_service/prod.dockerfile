FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Set timezone to avoid interactive prompt
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC

# Install Python 3.11
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3.11-distutils \
    python3-pip \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && adduser --disabled-password --gecos '' appuser

WORKDIR /app

# Set production environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONOPTIMIZE=2 \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH \
    CUDA_HOME=/usr/local/cuda

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies and attempt CUDA PyTorch upgrade
RUN pip install --no-cache-dir -r requirements.txt \
    && (pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu118 "torch==2.0.1+cu118" \
    && echo "Installed CUDA-enabled PyTorch (cu118).") \
    || echo "CUDA-enabled PyTorch wheel not available; using CPU-only PyTorch."

# Copy application code and set ownership
COPY . .
ENV STREAMLIT_CONFIG_FILE=/app/.streamlit/config.toml
RUN chown -R appuser:appuser /app

# Switch to non-root user for security
USER appuser

# Expose port
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
