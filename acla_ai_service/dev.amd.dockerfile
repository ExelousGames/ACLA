FROM rocm/dev-ubuntu-22.04:5.4.2

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
    git \
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
    HSA_OVERRIDE_GFX_VERSION=10.3.0

# Copy requirements first for better caching
COPY requirements.common.txt .
COPY requirements.amd.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.amd.txt

# Copy the rest of the application
COPY . .

# Expose ports
EXPOSE 8000 8501

# Start the application
CMD ["./start-dev.sh"]
