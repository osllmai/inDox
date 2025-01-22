# IndoxMiner

[![PyPI version](https://badge.fury.io/py/indoxminer.svg)](https://badge.fury.io/py/indoxminer)  
[![License: MIT](https://img.shields.io/badge/License-AGPL-yellow.svg)](https://opensource.org/licenses/AGPL)


IndoxMiner is a powerful Python library that leverages Large Language Models (LLMs) for **data extraction**, advanced **object detection**, and **image classification**. It combines schema-based data extraction from unstructured data sources such as text, PDFs, and images, with state-of-the-art object detection and image classification models. IndoxMiner enables seamless automation for document processing, visual recognition, and classification tasks.


## 🚀 Key Features

- **Data Extraction**: Extract structured data from text, PDFs, and images using schema-based extraction and LLMs.
- **Object Detection**: Leverage pre-trained object detection models for high-accuracy real-time image recognition.
- **Image Classification**: Utilize advanced classification models for identifying objects or features in images.
- **OCR Integration**: Extract text from scanned documents or images with integrated OCR engines.
- **Schema-Based Extraction**: Define custom schemas for data extraction with validation and type-safety.
- **Multi-Model Support**: Supports a wide range of models for detection and classification.
- **Async Support**: Built for scalability with asynchronous processing capabilities.
- **Flexible Outputs**: Export results to JSON, pandas DataFrames, or custom formats.

---

## 📦 Installation

You may also install required object detection dependencies like Detectron2 or YOLOv8 using:
Install IndoxMiner with:

```bash
pip install indoxminer
```

You may also install additional dependencies for object detection and classification, such as YOLOv8 or Detectron2.

---

## 📝 Quick Start

### 1. Data Extraction

IndoxMiner integrates seamlessly with OpenAI models for **schema-based extraction** from text, PDFs, and images. Here's how you can extract structured data from a document:

#### Basic Text Extraction

```python
from indoxminer import ExtractorSchema, Field, FieldType, ValidationRule, Extractor, OpenAi

# Initialize OpenAI extractor
llm_extractor = OpenAi(api_key="your-api-key", model="gpt-4-mini")

# Define extraction schema
schema = ExtractorSchema(
    fields=[
        Field(name="product_name", field_type=FieldType.STRING, rules=ValidationRule(min_length=2)),
        Field(name="price", field_type=FieldType.FLOAT, rules=ValidationRule(min_value=0))
    ]
)

# Create extractor and process text
extractor = Extractor(llm=llm_extractor, schema=schema)
text = """
MacBook Pro 16-inch with M2 chip
Price: $2,399.99
In stock: Yes
"""

result = await extractor.extract(text)
df = extractor.to_dataframe(result)
```

---

### 2. Object Detection

IndoxMiner provides powerful **object detection** capabilities with support for a variety of models, such as YOLO, Detectron2, and DETR.

#### Supported Models for Object Detection

| Model             | Supported ✅ |
|--------------------|:------------:|
| **Detectron2**     | ✅           |
| **DETR**           | ✅           |
| **DETR-CLIP**      | ✅           |
| **GroundingDINO**  | ✅           |
| **Grounded-SAM2**  | ✅           |
| **Grounded-SAM2-FLorence2**  | ✅           |
| **Kosmos2**        | ✅           |
| **OWL-ViT**        | ✅           |
| **OWL-V2**        | ✅           |
| **RT-DETR**        | ✅           |
| **SAM2**           | ✅           |
| **YOLOv5**         | ✅           |
| **YOLOv6**         | ✅           |
| **YOLOv7**         | ✅           |
| **YOLOv8**         | ✅           |
| **YOLOv10**        | ✅           |
| **YOLOv11**        | ✅           |
| **YOLOX**          | ✅           |


#### Example: Object Detection with YOLOv5

```python
from indoxminer.detection import YOLOv5

# Initialize YOLOv5 model
detector = YOLOv5()

# Detect objects in an image
image_path = "dog-cat-under-sheet.jpg"
outputs = await detector.detect_objects(image_path)


You can also switch to other models by specifying the model name, e.g., `detectron2`, `detr`, `yolov8`, etc.

```python
detector = YOLOv8()  # For YOLOv8

```

---

### 3. Image Classification

IndoxMiner now supports advanced **image classification** with models like SigCLIP, ViT, MetaCLIP, MobileCLIP, BioCLIP, AltCLIP, and RemoteCLIP.

#### Supported Models for Classification

| Model              | Description                          |
|--------------------|--------------------------------------|
| **SigCLIP**        | Semantic image classification model. |
| **ViT**            | Vision Transformer for image classification. |
| **MetaCLIP**       | Meta AI’s advanced CLIP model.       |
| **MobileCLIP**     | Mobile-optimized CLIP.               |
| **BioCLIP**        | Specialized for biological images.   |
| **AltCLIP**        | Alternative CLIP from BAAI.          |
| **RemoteCLIP**     | Remote sensing-specific CLIP model.  |

#### Example: Classification with RemoteCLIP

```python
from indoxminer.classification import RemoteCLIPClassifier
from PIL import Image

# Initialize RemoteCLIP
classifier = RemoteCLIPClassifier(
    model_name="ViT-L-14",
    checkpoint_path="/path/to/RemoteCLIP-ViT-L-14.pt"
)

# Classify an image
image = Image.open("/path/to/airport.jpg")
labels = ["An airport", "A university", "A stadium"]
classifier.classify(image, labels, top=3)
```

#### Example: Classification with SigCLIP

```python
from indoxminer.classification import SigCLIPClassifier
from PIL import Image

# Initialize SigCLIP
classifier = SigCLIPClassifier()

# Classify an image with default labels
image = Image.open("/path/to/image.jpg")
classifier.classify(image)
```

---

## 🔧 Core Components

### Classification Models

- **Flexible Input**: Supports single or batch image classification.
- **Custom Labels**: Define your own labels for classification tasks.
- **Visualization**: Generates bar plots of predicted probabilities.

---

## 🔍 Error Handling

IndoxMiner provides detailed error reporting for both data extraction, object detection, and classification tasks.

```python
try:
    results = await extractor.extract(documents)
except Exception as e:
    print(f"An error occurred: {e}")
```

---

## 🤝 Contributing

We welcome contributions! Please follow the standard workflow:
1. Fork the repository.
2. Create a feature branch.
3. Commit your changes.
4. Push to the branch.
5. Open a Pull Request.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🆘 Support

- **Documentation**: [Full documentation](https://indoxminer.readthedocs.io/)
- **Issues**: [GitHub Issues](https://github.com/username/indoxminer/issues)
- **Discussions**: [GitHub Discussions](https://github.com/username/indoxminer/discussions)


--- 

## 🌟 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=username/indoxminer&type=Date)](https://star-history.com/#username/indoxminer&Date)


