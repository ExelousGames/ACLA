FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies (CPU baseline)
RUN pip install --no-cache-dir -r requirements.txt

# Try to upgrade to CUDA-enabled PyTorch if compatible wheels are available.
# This will enable GPU usage inside the container when the host exposes NVIDIA GPUs.
# If the CUDA wheel for the pinned version/Python combo isn't available, this step
# will no-op and keep the CPU-only wheel installed from requirements.txt.
# CUDA 11.8 is broadly compatible with recent NVIDIA drivers and Docker setups.
RUN set -eux; \
    (pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu118 "torch==2.0.1+cu118" \
    && echo "Installed CUDA-enabled PyTorch (cu118)." ) \
    || echo "CUDA-enabled PyTorch wheel not available for this Python/arch; using CPU-only PyTorch."

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Command to run the application in development mode with hot reload
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
