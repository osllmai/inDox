# IndoxMiner - Object Detection

[![PyPI version](https://badge.fury.io/py/indoxminer.svg)](https://badge.fury.io/py/indoxminer)
[![License: MIT](https://img.shields.io/badge/License-AGPL-yellow.svg)](https://opensource.org/licenses/AGPL)

IndoxMiner offers powerful, pre-trained models for real-time and high-accuracy object detection. Whether you're working with images, videos, or other visual data sources, IndoxMiner provides state-of-the-art solutions from renowned models like **DETR**, **YOLO**, and **Detectron2**.

## 🚀 Key Features

- **Wide Model Support**: Utilize models such as Detectron2, DETR, GroundingDINO, YOLO, and more.
- **Seamless Integration**: No need to worry about model installation. IndoxMiner handles the dependencies for you!
- **Pretrained Weights**: All models come with pre-configured weights and checkpoints — just input them when initializing the model.
- **Auto-Installation**: If a required model or its dependencies are missing, IndoxMiner automatically installs them for you, making setup easy and hassle-free.

## 🎯 Supported Models

IndoxMiner supports the following cutting-edge models for object detection:

| Model              | Description                                                                 |
|--------------------|-----------------------------------------------------------------------------|
| [**Detectron2**](https://github.com/facebookresearch/detectron2)   | A fast and flexible object detection library by Facebook AI Research. |
| [**DETR**](https://github.com/facebook/detr)                     | DEtection TRansformers (DETR), a transformer-based object detection model. |
| [**DETR-CLIP**](https://github.com/facebook/detr)                 | Combines DETR with CLIP for improved performance in detecting a wide variety of objects. |
| [**GroundingDINO**](https://github.com/IDEA-Research/GroundingDINO) | Vision-language model for grounding objects in the visual scene. |
| [**Kosmos2**](https://github.com/microsoft/Kosmos)                  | A cross-modal vision-language pretraining model. |
| [**OWL-ViT**](https://github.com/facebookresearch/owl_vit)         | Open-Vocabulary Vision Transformer for universal object detection. |
| [**RT-DETR**](https://github.com/facebook/detr)                   | Real-Time DEtection TRansformers for fast and efficient object detection. |
| [**SAM2**](https://github.com/facebookresearch/sam)                | Segment Anything Model, enabling flexible and interactive image segmentation. |
| [**YOLOv5**](https://github.com/ultralytics/yolov5)                | YOLOv5, one of the fastest and most accurate real-time object detection models. |
| [**YOLOv6**](https://github.com/meituan/YOLOv6)                    | YOLOv6, a version optimized for edge devices and high-speed detection. |
| [**YOLOv7**](https://github.com/WongKinYiu/yolov7)                 | YOLOv7, an improved version of YOLO with better performance and accuracy. |
| [**YOLOv8**](https://github.com/ultralytics/yolov8)                | YOLOv8, the latest in the YOLO series with even more enhancements. |
| [**YOLOv10**](https://github.com/ultralytics/yolov5)               | A cutting-edge version of YOLO, designed for both speed and accuracy. |
| [**YOLOv11**](https://github.com/ultralytics/yolov5)               | YOLOv11, the most recent YOLO model with further optimizations. |
| [**YOLOX**](https://github.com/Megvii-BaseDetection/YOLOX)          | A highly optimized, scalable version of the YOLO family. |

---

## 🎯 Easy Setup and Usage

IndoxMiner takes care of everything for you. If you don't have the required models or dependencies installed, don't worry! The library will detect what's missing and automatically handle the installation process. Simply input the model name when initializing, and IndoxMiner will take care of the rest, including fetching the required pretrained weights and checkpoints.

### Example Usage

```python
from indoxminer.detection import DETR

# Initialize the DETR model with specific config and weights
detr_model = DETR(model="detr", weights="path/to/weights")

# Perform object detection on an image
outputs = detr_model.detect_objects(image_path="/path/to/image.jpg")

# Visualize detection results
detr_model.visualize_results(outputs)
```

### Model Initialization

To use any supported object detection model, simply specify the model name and provide the path to the checkpoint or weight file. Here's an example of how to initialize and run a model:

```python
# For Detectron2
from indoxminer.detection import Detectron2
model = Detectron2(weights="path/to/detectron2_checkpoint")

# For YOLOv5
from indoxminer.detection import YOLOv5
model = YOLOv5(weights="path/to/yolov5_checkpoint")
```

### Auto-Installation of Models

IndoxMiner ensures that you are always equipped with the latest model versions. If the required models are not installed, the library will automatically attempt to download and install the necessary dependencies for you.

## 🧑‍💻 Configuration Options

Each model comes with a set of customizable options for fine-tuning the detection process. Some common configuration options include:

- **Model**: The model name (e.g., `detectron2`, `yolov5`, `detr`)
- **Weights**: Path to pre-trained model weights or checkpoints
- **Confidence Threshold**: Set a confidence threshold for detection (default is 0.5)
- **Device**: The device to run the model on (`cpu` or `cuda` for GPU acceleration)

```python
model = YOLOv5(
    weights="path/to/yolov5_checkpoint", 
    confidence_threshold=0.6, 
    device="cuda"
)
```

## 🖼️ Visualizing Detection Results

IndoxMiner provides built-in functions to visualize the detection results. The bounding boxes, class labels, and confidence scores are drawn directly on the image, making it easy to interpret the results.

```python
model.visualize_results(outputs)  # Displays bounding boxes and labels
```

## 🤖 Supported Formats and Outputs

Detected objects are returned as a list of dictionaries with the following details:

- `bbox`: Bounding box coordinates (x1, y1, x2, y2)
- `class_id`: The class ID of the detected object
- `confidence`: Confidence score for the prediction

For example:

```python
{
    "bbox": [x1, y1, x2, y2],
    "class_id": 63,  # COCO class ID
    "confidence": 0.87
}
```

## 🚀 Model Weights and Pre-trained Checkpoints

All necessary model weights and checkpoints are included and can be accessed directly. You don't have to manually download or set up the weights; just point to the path and you're ready to go!

---

## 🤝 Contributing

We welcome contributions to improve and extend the capabilities of IndoxMiner. To contribute:

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to your branch
5. Open a Pull Request

For guidelines and more information, please refer to the [Contributing Guidelines](CONTRIBUTING.md).

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- **Documentation**: [Full documentation](https://indoxminer.readthedocs.io/)
- **Issues**: [GitHub Issues](https://github.com/username/indoxminer/issues)
- **Discussions**: [GitHub Discussions](https://github.com/username/indoxminer/discussions)
