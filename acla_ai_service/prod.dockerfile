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

# Attempt to install CUDA-enabled PyTorch to leverage GPUs when available.
# Falls back to CPU-only PyTorch if matching CUDA wheels aren't present for this env.
RUN set -eux; \
    (pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu118 "torch==2.0.1+cu118" \
    && echo "Installed CUDA-enabled PyTorch (cu118)." ) \
    || echo "CUDA-enabled PyTorch wheel not available; using CPU-only PyTorch."

# Copy application code
COPY . .

# Create non-root user for security
RUN adduser --disabled-password --gecos '' appuser && chown -R appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
