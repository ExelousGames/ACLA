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
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

WORKDIR /app

# Set all environment variables for optimization
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONOPTIMIZE=1 \
    OMP_NUM_THREADS=2 \
    MKL_NUM_THREADS=2 \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH \
    CUDA_HOME=/usr/local/cuda

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies in logical groups to optimize memory usage
RUN pip install --no-cache-dir \
    # Core FastAPI stack
    fastapi==0.104.1 \
    uvicorn==0.24.0 \
    pydantic==2.5.0 \
    pydantic-settings==2.1.0 \
    typing-extensions==4.8.0 \
    # HTTP and async utilities  
    requests==2.31.0 \
    python-multipart==0.0.6 \
    httpx==0.25.2 \
    aiofiles==23.2.1 \
    aiohttp==3.9.1 \
    python-dotenv==1.0.0 \
    && pip install --no-cache-dir \
    # Data science core
    pandas==2.1.4 \
    numpy==1.24.4 \
    scikit-learn==1.3.2 \
    hmmlearn==0.3.2 \
    joblib==1.3.2 \
    river==0.21.0 \
    && pip install --no-cache-dir \
    # PyTorch with CUDA support (separate for memory efficiency)
    torch==2.0.1 --index-url https://download.pytorch.org/whl/cu118 \
    && pip install --no-cache-dir \
    # Local LLM fine-tuning stack
    huggingface-hub==0.35.3 \
    transformers==4.37.2 \
    tokenizers==0.15.1 \
    peft==0.8.2 \
    accelerate==0.26.1 \
    bitsandbytes==0.42.0 \
    sentencepiece==0.1.99 \
    safetensors==0.4.2 \
    tensorboard==2.15.1 \
    datasets==2.16.1 \
    && pip install --no-cache-dir \
    # Visualization and data processing
    plotly==5.17.0 \
    matplotlib==3.8.2 \
    seaborn==0.13.0 \
    pyarrow==14.0.0 \
    zarr==2.16.1 \
    numcodecs==0.12.1 \
    lz4==4.3.2 \
    xxhash==3.4.1 \
    # Development and AI tools
    pytest==7.4.0 \
    pytest-asyncio==0.21.1 \
    openai==1.3.7 \
    streamlit==1.28.0  # provides the telemetry annotation UI

# Copy application code and setup in single layer
COPY . .
ENV STREAMLIT_CONFIG_FILE=/app/.streamlit/config.toml
RUN chmod +x /app/start-dev.sh

# Expose port
EXPOSE 8000

# Command to run the application in development mode with memory-efficient options
CMD ["/app/start-dev.sh", "no-reload"]
