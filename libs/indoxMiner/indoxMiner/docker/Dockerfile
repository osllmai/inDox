# Base image for Python
FROM python:3.9-slim-bullseye

# Install required system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    curl \
    git \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    ninja-build && \
    rm -rf /var/lib/apt/lists/*

# Create a virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create a non-root user and set permissions
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /opt/venv

# Set environment variables for caching and other paths
ENV HOME=/app \
    XDG_CACHE_HOME=/app/cache \
    MODEL_CACHE_DIR=/app/cache/models \
    TRANSFORMERS_CACHE=/app/cache/huggingface \
    MPLCONFIGDIR=/app/matplotlib-cache \
    TORCH_HOME=/app/cache/torch

# Set working directory
WORKDIR /app

# Copy dependency files first (to leverage Docker caching)
COPY requirements-base.txt /app/requirements-base.txt
RUN pip install --no-cache-dir -r /app/requirements-base.txt

# Copy application files
COPY . /app

# Prepare cache directories and set permissions
RUN mkdir -p /app/cache/huggingface /app/cache/torch /app/matplotlib-cache /app/cache/models && \
    chmod -R 777 /app/cache && \
    chown -R appuser:appuser /app

# Add entrypoint script and make it executable
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Switch to non-root user
USER appuser

# Set entry point
ENTRYPOINT ["/app/entrypoint.sh"]
