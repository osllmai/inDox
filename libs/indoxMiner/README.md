# IndoxMiner

[![PyPI version](https://badge.fury.io/py/indoxminer.svg)](https://badge.fury.io/py/indoxminer)  
[![License: MIT](https://img.shields.io/badge/License-AGPL-yellow.svg)](https://opensource.org/licenses/AGPL)

IndoxMiner is a powerful Python library that leverages Large Language Models (LLMs) for **data extraction**, advanced **object detection**, **image classification**, and **multimodal vision-language processing**. It combines schema-based data extraction from unstructured data sources such as text, PDFs, and images, with state-of-the-art object detection, image classification, and multimodal models. IndoxMiner enables seamless automation for document processing, visual recognition, classification tasks, and now **vision-language** understanding.

---

## 🚀 Key Features

- **Data Extraction**: Extract structured data from text, PDFs, and images using schema-based extraction and LLMs.  
- **Object Detection**: Leverage pre-trained object detection models for high-accuracy real-time image recognition.  
- **Image Classification**: Utilize advanced classification models (CLIP-based or Transformers) for identifying objects or features in images.  
- **Multimodal Models**: Integrate vision-language models like BLIP-2 and LLaVA-NeXT for image captioning and image-based Q&A.  
- **OCR Integration**: Extract text from scanned documents or images with integrated OCR engines.  
- **Schema-Based Extraction**: Define custom schemas for data extraction with validation and type-safety.  
- **Multi-Model Support**: Supports a wide range of models for detection and classification.  
- **Async Support**: Built for scalability with asynchronous processing capabilities.  
- **Flexible Outputs**: Export results to JSON, pandas DataFrames, or custom formats.

---

## 📦 Installation

Install IndoxMiner with:

```bash
pip install indoxminer
```

You may also install additional dependencies for:
- **Object Detection** (e.g., YOLOv8, Detectron2)
- **Image Classification** (e.g., CLIP-based models)
- **Multimodal** (e.g., LLaVA, BLIP-2)

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

| Model                             | Supported? |
|-----------------------------------|:---------:|
| **Detectron2**                    | ✅        |
| **DETR**                          | ✅        |
| **DETR-CLIP**                     | ✅        |
| **GroundingDINO**                 | ✅        |
| **Grounded-SAM2**                 | ✅        |
| **Grounded-SAM2-FLorence2**       | ✅        |
| **Kosmos2**                       | ✅        |
| **OWL-ViT**                       | ✅        |
| **OWL-V2**                        | ✅        |
| **RT-DETR**                       | ✅        |
| **SAM2**                          | ✅        |
| **YOLOv5**                        | ✅        |
| **YOLOv6**                        | ✅        |
| **YOLOv7**                        | ✅        |
| **YOLOv8**                        | ✅        |
| **YOLOv10**                       | ✅        |
| **YOLOv11**                       | ✅        |
| **YOLOX**                         | ✅        |

#### Example: Object Detection with YOLOv5

```python
from indoxminer.detection import YOLOv5

# Initialize YOLOv5 model
detector = YOLOv5()

# Detect objects in an image
image_path = "dog-cat-under-sheet.jpg"
outputs = await detector.detect_objects(image_path)
```

You can also switch to other models by specifying the model name, e.g., `detectron2`, `detr`, `yolov8`, etc.

```python
detector = YOLOv8()  # For YOLOv8
```

---

### 3. Image Classification

IndoxMiner’s **classification** module supports a variety of CLIP-based and Transformer-based models for **image classification**. Below is a table of currently available classifiers found in `indoxminer/classification/`.

| Classifier             | Python File                      | Description                                                                                |
|------------------------|----------------------------------|--------------------------------------------------------------------------------------------|
| **CLIP**     | `clip.py`             | Base CLIP model for general-purpose image classification.                                  |
| **SigLIP**  | `SigLIP.py`          | a vision-language model that uses a sigmoid-based contrastive learning objective to compute text-image similarity scores.               |
| **ViT**      | `vit.py`              | Vision Transformer (ViT) for image classification.                                         |
| **AltCLIP**  | `altclip.py`          | Alternative CLIP from BAAI.                                                                |
| **BioCLIP**  | `bioclip.py`          | Specialized CLIP model for biological images.                                              |
| **BioMedCLIP** | `biomedclip.py`     | Specialized CLIP model for biomedical or medical imaging tasks.                            |
| **MetaCLIP** | `metaclip.py`         | Meta AI’s advanced CLIP model.                                                             |
| **MobileCLIP** | `mobileclip.py`     | Mobile-optimized CLIP for on-device image classification.                                  |
| **RemoteCLIP** | `remoteclip.py`     | Remote sensing–specific CLIP model for satellite or aerial imagery.                        |

#### Example: Classification with RemoteCLIP

```python
from indoxminer.classification import RemoteCLIP
from PIL import Image

# Initialize RemoteCLIP
classifier = RemoteCLIP(
    model_name="ViT-L-14",
    checkpoint_path="/path/to/RemoteCLIP-ViT-L-14.pt"
)

# Classify an image
image = Image.open("/path/to/airport.jpg")
labels = ["Airport", "University", "Stadium"]
results = classifier.classify(image, labels, top=3)

print(results)
```

#### Example: Classification with SigLIP

```python
from indoxminer.classification import SigLIP
from PIL import Image

# Initialize SigLIP
classifier = SigLIP()

# Classify an image with default (or custom) labels
image = Image.open("/path/to/image.jpg")
results = classifier.classify(image, ["cat", "dog", "bird"], top=1)

print(results)
```

---

### 4. Multimodal Models

In addition to object detection and classification, IndoxMiner also supports **vision-language (multimodal)** models that process both images and text. This enables **natural language understanding** of images, image captioning, and image-based question answering.

#### Supported Multimodal Models
- **LLaVA-NeXT** (LLaVA + LLaMA 3 8B Instruct)
- **BLIP-2** (Vision-Language Model for Captioning & Question Answering)

These models allow users to ask questions about images and receive **detailed natural language responses** or generate captions automatically.

##### Installation (for LLaVA Model)

1. **Clone and install LLaVA-NeXT**:

   ```bash
   git clone https://github.com/LLaVA-VL/LLaVA-NeXT
   cd LLaVA-NeXT
   pip install -e .
   ```

2. **Modify LLaVA to Use LLaMA 3**  
   LLaVA uses a default model, but for better performance, update `conversation.py` to reference **`NousResearch/Meta-Llama-3-8B-Instruct`**:

   ```python
   tokenizer_id="NousResearch/Meta-Llama-3-8B-Instruct"
   ```

---

#### Usage

##### Using LLaVA Model

```python
from indoxminer.multimodal.llava_model import LLaVAModel

# Initialize the model
model = LLaVAModel()

# Provide a local image path and a question
image_path = "path/to/your/image.jpg"
question = "What is shown in this image?"

# Generate response
response = model.generate_response(image_path, question)
print("LLaVA Response:", response)
```

##### Using BLIP-2 Model

```python
from indoxminer.multimodal.blip2_model import BLIP2Model

# Initialize the model
model = BLIP2Model()

# Provide a local image path
image_path = "path/to/your/image.jpg"

# Generate a caption
caption = model.generate_response(image_path)
print("Generated Caption:", caption)

# Ask a question about the image
question = "How many objects are there?"
answer = model.generate_response(image_path, question)
print("Model Answer:", answer)
```

---

## 🔧 Core Components

### Classification Models

- **Flexible Input**: Supports single or batch image classification.  
- **Custom Labels**: Define your own labels for classification tasks.  
- **Visualization**: (Optional) Generate bar plots or logs of predicted probabilities.

### Multimodal Models

- **Vision-Language Reasoning**: Enables natural-language Q&A about images.  
- **Image Captioning**: Automatically generate descriptive captions for images.  
- **Extendable**: Additional models like Kosmos-2 or GPT-4V can be added.

---

## 🔍 Error Handling

IndoxMiner provides detailed error reporting for all of its modules—data extraction, object detection, classification, and multimodal tasks.

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

---

**Happy coding with IndoxMiner!**