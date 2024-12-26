# Use CUDA-enabled base image
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

LABEL name="indoxminer-detr"

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libomp-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV MODEL_CHECKPOINT="facebook/detr-resnet-50"

CMD ["python", "infrence.py"]