# Use the base image
FROM indoxminer-base:latest

# Set environment variables
ENV FORCE_CUDA=0 \
    TORCH_CUDA_ARCH_LIST="None" \
    PYTHONPATH="/app:${PYTHONPATH:-}"

# Switch to root to install system packages
USER root

# Retry logic for apt and multiple mirrors
RUN echo 'Acquire::Retries "3";' > /etc/apt/apt.conf.d/80-retries && \
    echo "deb http://deb.debian.org/debian bullseye main\ndeb http://security.debian.org/debian-security bullseye-security main\ndeb http://deb.debian.org/debian bullseye-updates main" > /etc/apt/sources.list

# Install required system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsm6 \
    libxext6 \
    git \
    python3-dev \
    gcc \
    g++ \
    curl \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install PyTorch and torchvision
RUN pip install --no-cache-dir torch==2.1.0 torchvision==0.16.0

# Copy the model's requirements.txt file
COPY --chown=appuser:appuser models/groundedsam2florence2/requirements.txt /app/

# Install Python dependencies from requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Download and install Grounded SAM2 with retry logic
RUN for i in $(seq 1 3); do \
        echo "Attempt $i: Downloading and installing Grounded SAM2" && \
        curl -L --max-time 300 https://github.com/IDEA-Research/Grounded-SAM-2/archive/refs/heads/main.tar.gz -o grounded-sam2.tar.gz && \
        tar xzf grounded-sam2.tar.gz && \
        cd Grounded-SAM-2-main && \
        pip install --no-cache-dir . && \
        cd .. && \
        rm -rf Grounded-SAM-2-main grounded-sam2.tar.gz && \
        break || \
        if [ $i -eq 3 ]; then exit 1; fi; \
        sleep 10; \
    done

# Switch back to non-root user
USER appuser

# Copy the model code to the working directory
COPY --chown=appuser:appuser models/groundedsam2florence2 /app/app/models/groundedsam2florence2

# Ensure the model folder is a Python package
RUN touch /app/app/models/__init__.py
RUN touch /app/app/models/groundedsam2florence2/__init__.py

# Set the working directory
WORKDIR /app

# Expose port
EXPOSE 8000

# Start the FastAPI app
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]