# IndoxMiner

[![PyPI version](https://badge.fury.io/py/indoxminer.svg)](https://badge.fury.io/py/indoxminer)
[![License: MIT](https://img.shields.io/badge/License-AGPL-yellow.svg)](https://opensource.org/licenses/AGPL)

IndoxMiner is a powerful Python library that leverages Large Language Models (LLMs) for **data extraction** and advanced **object detection**. It combines schema-based data extraction from unstructured data sources such as text, PDFs, and images, with state-of-the-art object detection models. IndoxMiner enables seamless automation for document processing and visual recognition tasks.

## 🚀 Key Features

- **Multi-Format Data Extraction**: Extract structured data from text, PDFs, images, and scanned documents.
- **Schema-Based Extraction**: Define custom schemas to extract precise data, ensuring data integrity and quality.
- **Object Detection Models**: Integrate a wide range of pre-trained models for object detection in images and videos, including YOLO, Detectron2, and more.
- **LLM Integration**: Intelligent data extraction powered by OpenAI models.
- **Validation & Type Safety**: Built-in validation rules and type-safe field definitions to ensure accurate data extraction.
- **Flexible Output**: Export extraction results to JSON, pandas DataFrames, or custom formats.
- **Async Support**: Built for scalability with asynchronous processing capabilities.
- **OCR Integration**: Options for OCR engines such as EasyOCR, Tesseract, and PaddleOCR for image-based text extraction.
- **High-Resolution Support**: Enhanced PDF processing for high-resolution documents.
- **Error Handling**: Comprehensive error handling and validation reporting for data extraction and detection tasks.

## 📦 Installation

To install the latest version of IndoxMiner, use pip:

```bash
pip install indoxminer
```

You may also install required object detection dependencies like Detectron2 or YOLOv8 using:

```bash
pip install 'git+https://github.com/facebookresearch/detectron2.git'
pip install ultralytics
```

---

## 📝 Data Extraction

IndoxMiner allows you to extract structured data from various formats like text, PDFs, and images using **schema-based extraction** and integration with powerful language models (LLMs).

### Basic Text Extraction

```python
from indoxminer import ExtractorSchema, Field, FieldType, ValidationRule, Extractor, OpenAi

# Initialize OpenAI extractor
llm_extractor = OpenAi(
    api_key="your-api-key",
    model="gpt-4-mini"
)

# Define extraction schema
schema = ExtractorSchema(
    fields=[
        Field(
            name="product_name",
            description="Product name",
            field_type=FieldType.STRING,
            rules=ValidationRule(min_length=2)
        ),
        Field(
            name="price",
            description="Price in USD",
            field_type=FieldType.FLOAT,
            rules=ValidationRule(min_value=0)
        )
    ]
)

# Create extractor and process text
extractor = Extractor(llm=llm_extractor, schema=schema)
text = """
MacBook Pro 16-inch with M2 chip
Price: $2,399.99
In stock: Yes
"""

# Extract and convert to DataFrame
result = await extractor.extract(text)
df = extractor.to_dataframe(result)
```

### PDF Processing

```python
from indoxminer import DocumentProcessor, ProcessingConfig

# Initialize processor with custom config
processor = DocumentProcessor(
    files=["invoice.pdf"],
    config=ProcessingConfig(
        hi_res_pdf=True,
        chunk_size=1000
    )
)

# Process document
documents = processor.process()

# Extract structured data
schema = ExtractorSchema(
    fields=[
        Field(
            name="bill_to",
            description="Billing address",
            field_type=FieldType.STRING
        ),
        Field(
            name="invoice_date",
            description="Invoice date",
            field_type=FieldType.DATE
        ),
        Field(
            name="total_amount",
            description="Total amount in USD",
            field_type=FieldType.FLOAT
        )
    ]
)

results = await extractor.extract(documents)
```

### Image Processing with OCR

```python
# Configure OCR-enabled processor
config = ProcessingConfig(
    ocr_enabled=True,
    ocr_engine="easyocr",  # or "tesseract", "paddle"
    language="en"
)

processor = DocumentProcessor(
    files=["receipt.jpg"],
    config=config
)

# Process image and extract text
documents = processor.process()
```

---

## 📷 Object Detection

IndoxMiner includes powerful object detection capabilities using pre-trained models. The library supports a wide range of models suitable for real-time and high-accuracy object detection tasks.

### Supported Detection Models

- **Detectron2** (from Facebook AI Research)
- **DETR** (DEtection TRansformers)
- **DETR-CLIP** (Combining DETR with CLIP for improved performance)
- **GroundingDINO** (Grounding vision-language models for better contextual understanding)
- **Kosmos2** (Cross-modal vision-language pre-training)
- **OWL-ViT** (Open-Vocabulary Vision Transformer for universal object detection)
- **RT-DETR** (Real-Time DEtection TRansformers)
- **SAM2** (Segment Anything Model for interactive image segmentation)
- **YOLOv5** (You Only Look Once model for real-time detection)
- **YOLOv6, YOLOv7, YOLOv8, YOLOv10, YOLOv11** (Updated versions of the popular YOLO object detection models)
- **YOLOX** (A robust, scalable version of the YOLO family)

These models are optimized for speed and accuracy, providing precise bounding boxes, class labels, and confidence scores for various objects in the image.

### 🚀 Quick Start - Object Detection

Here's a guide to using the IndoxMiner object detection models. This example demonstrates using the **YOLOv5** model for object detection:

#### Using YOLOv5 for Object Detection

```python
from indoxminer import ObjectDetection

# Initialize YOLOv5 detector
detector = ObjectDetection(model="yolov5")

# Run object detection on an image
image_path = "image.jpg"
detections = await detector.detect_objects(image_path)

# Visualize results
detector.visualize_results(detections)

# Optionally, save the detected image
detector.save_results(detections, "output.jpg")
```

#### Supported Detection Models

The `ObjectDetection` class in IndoxMiner can use any of the following models:

```python
detector = ObjectDetection(model="detectron2")  # for Detectron2
detector = ObjectDetection(model="detr")        # for DETR
detector = ObjectDetection(model="yolov8")      # for YOLOv8
```

You can also set additional configuration options such as confidence threshold, device (CPU/GPU), etc.

### 📊 Visualizing Detection Results

IndoxMiner provides simple methods to visualize detection results, such as bounding boxes, class labels, and confidence scores. The `visualize_results()` method displays the image with the bounding boxes drawn around the detected objects.

```python
detector.visualize_results(detections)  # Display bounding boxes and labels
```

### ⚙️ Configuration Options

You can configure various parameters of the object detection models for improved accuracy and performance:

- **model**: The model name (`"yolov5"`, `"detectron2"`, `"detr"`, etc.)
- **confidence_threshold**: Confidence threshold for object detection (default: 0.5)
- **device**: The device to run the model on (`"cpu"` or `"cuda"` for GPU acceleration)

Example:

```python
detector = ObjectDetection(
    model="yolov5", 
    confidence_threshold=0.6, 
    device="cuda"
)
```

### 💡 Supported Formats and Output

The detected objects are returned as a list of dictionaries with the following information:

- `bbox`: The bounding box coordinates (x1, y1, x2, y2)
- `class_id`: The predicted class ID (e.g., for COCO dataset)
- `confidence`: The confidence score of the prediction

For example:

```python
{
    "bbox": [x1, y1, x2, y2],
    "class_id": 63,
    "confidence": 0.87
}
```

### 🔄 Models and Pre-trained Weights

The available models and their pre-trained weights are downloaded automatically when you initialize the detector. This ensures that you always have access to the latest model versions without additional setup.

---

## 🔧 Core Components

### ExtractorSchema

Defines the structure of data to be extracted:

- Field definitions
- Validation rules
- Output format specifications

```python
schema = ExtractorSchema(
    fields=[...],
    output_format="json"
)
```

### Field Types

Supported field types:

- `STRING`: Text data
- `INTEGER`: Whole numbers
- `FLOAT`: Decimal numbers
- `DATE`: Date values
- `BOOLEAN`: True/False values
- `LIST`: Arrays of values
- `DICT`: Nested objects

### Validation Rules

Available validation options:

- `min_length`/`max_length`: String length constraints
- `min_value`/`max_value`: Numeric bounds
- `pattern`: Regex patterns
- `required`: Required fields
- `custom`: Custom validation functions

##

 ⚙️ Configuration Options

### ProcessingConfig

```python
config = ProcessingConfig(
    hi_res_pdf=True,          # High-resolution PDF processing
    ocr_enabled=True,         # Enable OCR
    ocr_engine="tesseract",   # OCR engine selection
    chunk_size=1000,          # Text chunk size
    language="en",            # Processing language
    max_threads=4             # Parallel processing threads
)
```

## 🔍 Error Handling

IndoxMiner provides detailed error reporting:

```python
results = await extractor.extract(documents)

if not results.is_valid:
    for chunk_idx, errors in results.validation_errors.items():
        print(f"Errors in chunk {chunk_idx}:")
        for error in errors:
            print(f"- {error.field}: {error.message}")

# Access valid results
valid_data = results.get_valid_results()
```

## 🤝 Contributing

We welcome contributions! To contribute:

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

Please read our [Contributing Guidelines](CONTRIBUTING.md) for more details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- **Documentation**: [Full documentation](https://indoxminer.readthedocs.io/)
- **Issues**: [GitHub Issues](https://github.com/username/indoxminer/issues)
- **Discussions**: [GitHub Discussions](https://github.com/username/indoxminer/discussions)

## 🌟 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=username/indoxminer&type=Date)](https://star-history.com/#username/indoxminer&Date)
```
