# IndoxMiner Docker Deployment

IndoxMiner is a core part of the Indox ecosystem, designed to efficiently handle **object detection** and **data extraction** tasks using various state-of-the-art models. This guide covers how to deploy IndoxMiner using Docker, including running multiple object detection models such as **DETR**, **Detectron2**, **YOLOv5**, and more, in a containerized environment.

---

## 📋 Prerequisites

- **Docker**: Ensure you have Docker and Docker Compose installed.
- **NVIDIA Container Toolkit**: For GPU acceleration.
- **GPU Support**: Required for models that use CUDA.
- **Data Folder**: Place the files you want to process in the `data` folder.

---

## 🏗️ Docker Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/osllmai/inDox.git
cd inDox/libs/indoxMiner/indoxMiner/docker
```

### 2️⃣ Build the Base Image

The base image is a foundational layer used by all model-specific Dockerfiles. Start by building the base image.

```bash
docker build -t indoxminer-base -f Dockerfile.base .
```

### 3️⃣ Build and Run Each Model Service

After the base image is built, proceed to build and run each model service one by one.

#### 🛠 Build DETR Model

```bash
docker build -t indoxminer-detr -f models/detr/detr.dockerfile .
docker run -d --name detr -p 8001:8000 --gpus all indoxminer-detr
```

#### 🛠 Build Detectron2 Model

```bash
docker build -t indoxminer-detectron2 -f models/detectron2/detectron2.dockerfile .
docker run -d --name detectron2 -p 8002:8000 --gpus all indoxminer-detectron2
```

#### 🛠 Build DETRCLIP Model

```bash
docker build -t indoxminer-detrclip -f models/detrclip/detrclip.dockerfile .
docker run -d --name detrclip -p 8003:8000 --gpus all indoxminer-detrclip
```

#### 🛠 Build GroundingDINO Model

```bash
docker build -t indoxminer-groundingdino -f models/groundingdino/groundingdino.dockerfile .
docker run -d --name groundingdino -p 8004:8000 --gpus all indoxminer-groundingdino
```

#### 🛠 Build Kosmos2 Model

```bash
docker build -t indoxminer-kosmos2 -f models/kosmos2/kosmos2.dockerfile .
docker run -d --name kosmos2 -p 8005:8000 --gpus all indoxminer-kosmos2
```

Repeat the process for other models:

| Model             | Build Command                                                            | Run Command                                                             |
| ----------------- | ------------------------------------------------------------------------ | ----------------------------------------------------------------------- |
| OWLVit            | `docker build -t indoxminer-owlvit -f models/owlvit/owlvit.dockerfile .` | `docker run -d --name owlvit -p 8006:8000 --gpus all indoxminer-owlvit` |
| RT-DETR           | `docker build -t indoxminer-rtdetr -f models/rtdetr/rtdetr.dockerfile .` | `docker run -d --name rtdetr -p 8007:8000 --gpus all indoxminer-rtdetr` |
| SAM2              | `docker build -t indoxminer-sam2 -f models/sam2/sam2.dockerfile .`       | `docker run -d --name sam2 -p 8008:8000 --gpus all indoxminer-sam2`     |
| YOLOv5 to YOLOv11 | Build and run similar to the above models.                               |

---

## ⚙️ Docker Services Overview

Each model runs as a separate service with its own Dockerfile and specific configurations.

---

## 🚀 Usage Instructions

### Accessing the Services

Each model runs on a different port. For example, to access the **DETR** model:

```bash
curl http://localhost:8001
```

To check the health status:

```bash
curl http://localhost:8001/health
```

### Sending an Image for Object Detection

Use the following command to send an image to the model:

```bash
curl -X POST http://localhost:8001/detect -F "image=@/path/to/your/image.jpg"
```

---

## 🔧 Port Mapping and User Manual

Here is a detailed guide on the ports used by each model in the Dockerized IndoxMiner application:

| Model                  | Port Mapping | Description                                   |
| ---------------------- | ------------ | --------------------------------------------- |
| DETR                   | 8001:8000    | Detection Transformers                        |
| Detectron2             | 8002:8000    | Facebook AI Research model                    |
| DETR-CLIP              | 8003:8000    | Enhanced with CLIP for better performance     |
| GroundingDINO          | 8004:8000    | Grounding vision-language model               |
| Kosmos2                | 8005:8000    | Cross-modal vision-language model             |
| OWLVit                 | 8006:8000    | Open-Vocabulary Vision Transformer            |
| RT-DETR                | 8007:8000    | Real-Time Detection Transformer               |
| SAM2                   | 8008:8000    | Segment Anything Model                        |
| YOLOv5                 | 8009:8000    | Real-time object detection                    |
| YOLOv6                 | 8010:8000    | Advanced YOLO version                         |
| YOLOv7                 | 8011:8000    | Enhanced accuracy and speed                   |
| YOLOv8                 | 8012:8000    | Latest YOLO version with optimizations        |
| YOLOv10                | 8013:8000    | Experimental YOLO version                     |
| YOLOv11                | 8014:8000    | Latest YOLO variant                           |
| YOLOX                  | 8015:8000    | Scalable YOLO model                           |
| GroundedSAM2           | 8016:8000    | Advanced segmentation model                   |
| GroundedSAM2 Florence2 | 8017:8000    | Enhanced version with better context handling |

### How to Use the Ports

- **Access the API Endpoints:**

  - Use `http://localhost:<port>` to access the service, replacing `<port>` with the respective port number.
  - For example, to access the **DETR** model:
    ```bash
    curl http://localhost:8001/detect -F "image=@/path/to/your/image.jpg"
    ```

- **Health Check:**
  - Use `http://localhost:<port>/health` to check if the service is running.
  - Example:
    ```bash
    curl http://localhost:8001/health
    ```

---

## 🔧 Configuration Options

You can customize the deployment by modifying the `docker-compose.yml` file:

| Configuration          | Description                      | Default Value |
| ---------------------- | -------------------------------- | ------------- |
| `MODEL_TYPE`           | The type of model to load        | N/A           |
| `MODEL_CACHE_DIR`      | Directory for caching models     | `/app/cache`  |
| `memory`               | Memory limit for each container  | 4G            |
| `gpu`                  | Number of GPUs to allocate       | 1             |
| `confidence_threshold` | Minimum confidence for detection | 0.5           |

---

## 🔄 Updating the Models

To update the models, rebuild the containers:

```bash
docker-compose down
docker-compose up --build
```

---

## 📄 License

IndoxMiner is licensed under the **AGPL License**. See the [LICENSE](LICENSE) file for details.

---

## 🆘 Support

For more help, contact us at **support@nerdstudio.ai** or visit our [website](https://nerdstudio.ai).

---

## 🤝 Contributing

We welcome contributions! To contribute:

1. Fork the repository.
2. Create a feature branch.
3. Commit your changes.
4. Push to the branch.
5. Open a Pull Request.

---

## 📖 Documentation

For full documentation, visit our [IndoxMiner Docs](https://nerdstudio.ai/docs/indoxminer).

---

## 🖥️ Using IndoxMiner as Part of the Indox Ecosystem

IndoxMiner is designed to be used as a modular component in the Indox ecosystem. Users can integrate the Dockerized object detection services into their existing workflows within Indox by:

1. **Cloning the Indox Repository:**

```bash
git clone https://github.com/osllmai/inDox.git
```

2. **Navigating to the Docker Directory:**

```bash
cd inDox/libs/indoxMiner/indoxMiner/docker
```

3. **Deploying the Models Using Docker Compose:**

```bash
docker-compose up --build
```

4. **Accessing the Object Detection Services:**
   - Use HTTP requests to interact with the models via their respective ports.
   - Integrate these services into the broader Indox applications for data extraction and processing.

The Indox ecosystem provides a unified platform for managing unstructured data, and IndoxMiner adds powerful object detection capabilities to the suite of tools available.
