FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install CUDA dependencies (if using GPU)
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

# Install Python packages
COPY requirements-base.txt .
RUN pip install --no-cache-dir -r requirements-base.txt

# Set Python path
ENV PYTHONPATH=/app

# Default command
CMD ["python3"]