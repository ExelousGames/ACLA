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

# Copy the rest of the application
COPY . .
RUN chmod +x /app/start-dev.sh \
    && mkdir -p /app/models/kokoro

# Frontend-facing API and chat WebSocket
EXPOSE 8000

# Start the application
CMD ["./start-dev.sh"]
