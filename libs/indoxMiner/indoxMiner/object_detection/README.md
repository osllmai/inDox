# Object Detection Models - IndoxMiner

This repository contains various pre-trained object detection models integrated into the IndoxMiner framework. These models provide a wide range of capabilities for object detection tasks and can be easily accessed through the IndoxMiner API.

## Available Models

### Transformer-Based Models
1. **RT-DETR**  
   A real-time object detection model based on DEtection TRansformers (DETR). Optimized for speed and performance.

2. **DETR**  
   The original DEtection TRansformers model designed for object detection with attention mechanisms.

3. **DETR-CLIP**  
   A variant of DETR that integrates CLIP (Contrastive Language–Image Pre-training) for improved zero-shot detection and multimodal understanding.

### YOLO Series
4. **YOLOX**  
   Builds upon the YOLO family with improved speed and accuracy.

5. **YOLOv5**  
   A popular and widely used YOLO version known for balance between speed and accuracy.

6. **YOLOv6**  
   Designed for better accuracy and speed in industrial applications.

7. **YOLOv7**  
   Offers state-of-the-art performance with architectural improvements over its predecessors.

8. **YOLOv8**  
   The latest YOLO iteration, focused on real-time performance.

9. **YOLOv10**  
   Early experimental model expanding YOLO's capabilities for diverse datasets.

10. **YOLOv11**  
    Advanced YOLO version with cutting-edge features and optimizations.

### Other Advanced Models
11. **Grounding DINO**  
    Combines dense prediction and transformers for grounding objects in an image.

12. **KOSMOS**  
    A multi-modal vision model for tasks including object detection.

13. **OWL-ViT**  
    A vision transformer leveraging pre-trained knowledge for general-purpose detection.

14. **Detectron2**  
    A modular framework for object detection and segmentation by Facebook AI Research (FAIR).

15. **SAM2**  
    Specializes in self-supervised learning, particularly effective for limited labeled data.

16. **LLAVANext**  
    A next-generation model supporting various inputs, from images to video frames.

---

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/osllmai/IndoxMiner.git
cd IndoxMiner
pip install -r requirements.txt
```

---

## Usage

Here’s an example of how to use the IndoxMiner object detection models:

```python
from indoxminer.object_detection import IndoxObjectDetection

# Specify the model name
model_name = "detr"  # Replace with the desired model, e.g., "yolov5", "groundingdino"
image_path = "/content/download.jpg"  # Replace with your image path

# Initialize the detector
indox_detector = IndoxObjectDetection(model_name=model_name, device="cuda")

# Run detection
indox_detector.run(image_path)
```

### Supported Models

Set `model_name` to any of the following:

- **Transformer-Based**: `"rtdetr"`, `"detr"`, `"detr-clip"`
- **YOLO Series**: `"yolox"`, `"yolov5"`, `"yolov6"`, `"yolov7"`, `"yolov8"`, `"yolov10"`, `"yolov11"`
- **Advanced Models**: `"groundingdino"`, `"kosmos"`, `"owlvit"`, `"detectron2"`, `"sam2"`, `"llavanext"`

---

## Notes for Users

1. Ensure GPU support (`device="cuda"`) for optimal performance.
2. Each model offers unique capabilities; select based on your task requirements.
3. Example queries for detection (if required):
   - `["a cat", "a dog", "a car"]`  
   Customize based on the objects you want to detect.

---

## Contributing

We welcome contributions!  
Feel free to suggest new models, improve documentation, or report issues via [GitHub](https://github.com/osllmai/IndoxMiner).

---

### Updates Log

- **Dec 2024**: Added new models: `detr`, `detr-clip`, `yolov5`, `yolov6`, `yolov7`, `yolov10`, `yolov11`.

---