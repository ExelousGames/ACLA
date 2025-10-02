FROM python:3.11-slim

WORKDIR /app

# Set production environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONOPTIMIZE=2

# Install system dependencies and create user in single layer
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* \
    && adduser --disabled-password --gecos '' appuser

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies and attempt CUDA PyTorch upgrade
RUN pip install --no-cache-dir -r requirements.txt \
    && (pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu118 "torch==2.0.1+cu118" \
    && echo "Installed CUDA-enabled PyTorch (cu118).") \
    || echo "CUDA-enabled PyTorch wheel not available; using CPU-only PyTorch."

# Copy application code and set ownership
COPY . .
RUN chown -R appuser:appuser /app

# Switch to non-root user for security
USER appuser

# Expose port
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
