FROM indoxminer-base:latest

USER root

# Retry logic for apt and multiple mirrors
RUN echo 'Acquire::Retries "3";' > /etc/apt/apt.conf.d/80-retries && \
    echo "deb http://deb.debian.org/debian bullseye main\ndeb http://security.debian.org/debian-security bullseye-security main\ndeb http://deb.debian.org/debian bullseye-updates main" > /etc/apt/sources.list

# Install system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsm6 \
    libxext6 \
    pkg-config && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV FORCE_CUDA=0 \
    TORCH_CUDA_ARCH_LIST="None" \
    PYTHONPATH="/app:${PYTHONPATH:-}"


# Install PyTorch and torchvision first, and make sure they're in the build environment
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch==2.1.0 torchvision==0.16.0 && \
    pip install --no-cache-dir setuptools wheel build

# Copy the requirements file
COPY models/detectron2/requirements.txt requirements.txt

# Create a requirements file without the git dependency
RUN grep -v "git+" requirements.txt > requirements_nogit.txt || true

# Install non-git requirements first
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --retries 3 --timeout 60 -r /app/requirements_nogit.txt

# Install detectron2 separately with build dependencies
RUN --mount=type=cache,target=/root/.cache/pip \
    PYTHONPATH="/usr/local/lib/python3.9/site-packages:$PYTHONPATH" \
    pip install --retries 3 --timeout 60 "git+https://github.com/facebookresearch/detectron2.git"

# Switch to appuser after installations are complete
USER appuser

# Copy model code
COPY --chown=appuser:appuser models/detectron2 /app/app/models/detectron2

# Make sure the models directory is a Python package
RUN touch /app/app/models/__init__.py
RUN touch /app/app/models/detectron2/__init__.py

# Set the working directory
WORKDIR /app


# Expose port
EXPOSE 8000

# Start the FastAPI app
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
