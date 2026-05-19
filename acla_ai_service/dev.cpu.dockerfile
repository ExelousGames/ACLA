FROM ubuntu:22.04

# Set timezone to avoid interactive prompt
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC

# Install Python 3.11
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    gnupg \
    curl \
    ca-certificates \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3.11-distutils \
    python3-pip \
    python3-venv \
    gcc \
    g++ \
    git \
    cmake \
    ninja-build \
    curl \
    bash \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Install Node.js 20 + Claude Code CLI (driven by claude-agent-sdk for the
# Claude annotation backend). Auth is supplied at runtime by bind-mounting
# the host's ~/.claude into the container — no API key is baked into the image.
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && npm install -g @anthropic-ai/claude-code \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Set all environment variables for optimization
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONOPTIMIZE=1

# Copy requirements first for better caching
COPY requirements.common.txt .
COPY requirements.cpu.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.cpu.txt
RUN pip install --no-cache-dir llama-cpp-python

# Copy application code and setup in single layer
COPY . .
ENV STREAMLIT_CONFIG_FILE=/app/.streamlit/config.toml
# CPU-only inference: by default offload zero layers to the (nonexistent) GPU
ENV LLAMA_N_GPU_LAYERS=0
RUN chmod +x /app/start-dev.sh /app/scripts/start_llama_server.sh \
    && mkdir -p /app/models/llama_server /app/models/kokoro

# Expose ports: 8000 = FastAPI, 8080 = llama-server, 8501 = streamlit UI
EXPOSE 8000 8080 8501

# Command to run the application in development mode with memory-efficient options
CMD ["/app/start-dev.sh"]
