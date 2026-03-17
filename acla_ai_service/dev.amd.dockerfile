FROM rocm/dev-ubuntu-24.04:6.4.4-complete

# Set timezone to avoid interactive prompt
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC

# Install Python 3.11 from source (Launchpad PPA is down)
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    git \
    build-essential \
    wget \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libxml2-dev \
    libxmlsec1-dev \
    libffi-dev \
    liblzma-dev \
    && wget https://www.python.org/ftp/python/3.11.9/Python-3.11.9.tgz \
    && tar xzf Python-3.11.9.tgz \
    && cd Python-3.11.9 \
    && ./configure --enable-optimizations \
    && make -j$(nproc) \
    && make altinstall \
    && cd .. \
    && rm -rf Python-3.11.9 Python-3.11.9.tgz \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Setup Virtual Environment
ENV VIRTUAL_ENV=/opt/venv
RUN python3.11 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

WORKDIR /app

# Upgrade pip inside the virtual environment
RUN pip install --upgrade pip

# Set all environment variables for optimization
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONOPTIMIZE=1 \
    OMP_NUM_THREADS=2 \
    MKL_NUM_THREADS=2 \
    HSA_OVERRIDE_GFX_VERSION=11.0.0

# Copy requirements first for better caching
COPY requirements.amd.txt .
COPY requirements.common.txt .

# Install PyTorch with ROCm support first
RUN pip install --no-cache-dir -r requirements.amd.txt

# Install other dependencies
RUN pip install --no-cache-dir -r requirements.common.txt
RUN CMAKE_ARGS="-DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1100" CC=/opt/rocm/llvm/bin/clang CXX=/opt/rocm/llvm/bin/clang++ pip install --no-cache-dir llama-cpp-python

# Copy the rest of the application
COPY . .

# Expose ports
EXPOSE 8000 8501

# Start the application
CMD ["./start-dev.sh"]
