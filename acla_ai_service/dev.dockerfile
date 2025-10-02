FROM python:3.11-slim

WORKDIR /app

# Set all environment variables for optimization
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONOPTIMIZE=1 \
    OMP_NUM_THREADS=2 \
    MKL_NUM_THREADS=2

# Install system dependencies and clean up in single layer
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

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
    joblib==1.3.2 \
    river==0.21.0 \
    && pip install --no-cache-dir \
    # PyTorch (separate for memory efficiency)
    torch==2.0.1 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir \
    # Visualization and data processing
    plotly==5.17.0 \
    matplotlib==3.8.2 \
    seaborn==0.13.0 \
    pyarrow==14.0.0 \
    lz4==4.3.2 \
    xxhash==3.4.1 \
    # Development and AI tools
    pytest==7.4.0 \
    pytest-asyncio==0.21.1 \
    openai==1.3.7

# Copy application code and setup in single layer
COPY . .
RUN chmod +x /app/start-dev.sh

# Expose port
EXPOSE 8000

# Command to run the application in development mode with memory-efficient options
CMD ["/app/start-dev.sh", "no-reload"]
