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

# Runtime code is the shared model/feature library; training owns the entrypoint.
COPY acla_ai_service/requirements*.txt /acla_ai_service/

# Copy requirements first for better caching
COPY acla_ai_training/requirements.amd.txt .
COPY acla_ai_training/requirements.common.txt .
COPY acla_ai_training/training/image_segmentation/requirements-labelme.txt training/image_segmentation/

# Install Python dependencies. The base image already provides Python 3.11,
# ROCm 7.2, and the matching PyTorch 2.10 stack.
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.common.txt \
    && pip install --no-cache-dir -r requirements.amd.txt

# Install tools for the Claude annotation backend.
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    ca-certificates \
    bash \
    libgl1 \
    libglib2.0-0 \
    tigervnc-standalone-server \
    openbox \
    novnc \
    fonts-dejavu-core \
    libxkbcommon-x11-0 \
    libxcb-xinerama0 \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-render-util0 \
    libxcb-shape0 \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js 20 + Claude Code CLI (driven by claude-agent-sdk for the
# Claude annotation backend). Auth is supplied at runtime by bind-mounting
# the host's ~/.claude into the container — no API key is baked into the image.
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && npm install -g @anthropic-ai/claude-code \
    && rm -rf /var/lib/apt/lists/*

# Copy the rest of the application
COPY acla_ai_service/ /acla_ai_service/
COPY acla_ai_training/ .
ENV PYTHONPATH=/app:/acla_ai_service
RUN chmod +x /app/start-dev.sh

# Local training UI and Labelme browser desktop
EXPOSE 8501 6080

# Start the application
CMD ["./start-dev.sh"]
