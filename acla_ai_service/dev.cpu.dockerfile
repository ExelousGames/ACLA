FROM ubuntu:22.04

# Set timezone to avoid interactive prompt
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC

# Install Python 3.11
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    gnupg \
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
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

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
RUN chmod +x /app/start-dev.sh

# Expose port
EXPOSE 8000 8501

# Command to run the application in development mode with memory-efficient options
CMD ["/app/start-dev.sh"]
