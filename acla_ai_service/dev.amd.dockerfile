FROM rocm/pytorch:rocm7.2_ubuntu22.04_py3.11_pytorch_release_2.10.0

# Set timezone to avoid interactive prompt
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC

WORKDIR /app

# Set all environment variables for optimization
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONOPTIMIZE=1

# Copy requirements first for better caching
COPY requirements.amd.txt .
COPY requirements.common.txt .

# Install Python dependencies. The base image already provides Python 3.11,
# ROCm 7.2, and the matching PyTorch 2.10 stack.
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.common.txt \
    && pip install --no-cache-dir -r requirements.amd.txt

# Install native llama.cpp with ROCm support
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    cmake \
    libcurl4-openssl-dev \
    && rm -rf /var/lib/apt/lists/*
RUN git clone https://github.com/ROCm/llama.cpp /opt/llama.cpp \
    && cd /opt/llama.cpp \
    && HIPCXX="$(hipconfig -l)/clang" HIP_PATH="$(hipconfig -R)" \
       cmake -S . -B build -DGGML_HIP=ON -DAMDGPU_TARGETS='gfx1151' \
       -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=ON \
    && cmake --build build --config Release -j$(nproc) \
    && ln -s /opt/llama.cpp/build/bin/llama-server /usr/local/bin/llama-server \
    && ln -s /opt/llama.cpp/build/bin/llama-cli /usr/local/bin/llama-cli

# Copy the rest of the application
COPY . .

# Expose ports
EXPOSE 8000 8501

# Start the application
CMD ["./start-dev.sh"]
