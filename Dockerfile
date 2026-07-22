# ==============================================================================
# Zenemoo AI - Production Dockerfile
# Multi-stage image build with PyTorch & OpenCV dependencies
# ==============================================================================

FROM python:3.11-slim as base

# Set Environment Variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install System Dependencies for OpenCV, PyTorch, and C Extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy Dependency Specifications
COPY requirements.txt .

# Install Python Packages
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

# Copy Application Source Code
COPY . .

# Ensure storage directories exist
RUN mkdir -p uploads outputs temp shared/weights static logs

# Create Non-Root User for Security
RUN useradd -m -u 1000 zenemoo && \
    chown -R zenemoo:zenemoo /app
USER zenemoo

# Expose FastAPI REST API Port
EXPOSE 8000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default Container Command (Launches FastAPI REST API Server)
CMD ["python", "main.py", "--api"]
