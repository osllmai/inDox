# IndoxMiner

[![PyPI version](https://badge.fury.io/py/indoxminer.svg)](https://badge.fury.io/py/indoxminer)  
[![License: MIT](https://img.shields.io/badge/License-AGPL-yellow.svg)](https://opensource.org/licenses/AGPL)

IndoxMiner is a powerful Python library that integrates **Large Language Models (LLMs)** for **data extraction** and cutting-edge **object detection**. Whether you're working with unstructured data such as text, PDFs, or images, or performing object detection with pretrained models, IndoxMiner streamlines your workflows with flexibility and high accuracy.

## 🚀 Key Features

- **Data Extraction**: Extract structured data from text, PDFs, and images using schema-based extraction and LLMs.
- **Object Detection**: Leverage pre-trained object detection models for high-accuracy real-time image recognition.
- **OCR Integration**: Extract text from scanned documents or images with integrated OCR engines.
- **Schema-Based Extraction**: Define custom schemas for data extraction with validation and type-safety.
- **Multi-Model Support**: Supports a wide range of object detection models including YOLO, DETR, and more.
- **Async Support**: Built for scalability with asynchronous processing capabilities.
- **Flexible Outputs**: Export results to JSON, pandas DataFrames, or custom formats.

---

## 📦 Installation

Install IndoxMiner with:

```bash
pip install indoxminer
```

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

#### PDF Processing

```python
from indoxminer import DocumentProcessor, ProcessingConfig

processor = DocumentProcessor(
    files=["invoice.pdf"],
    config=ProcessingConfig(hi_res_pdf=True, chunk_size=1000)
)

documents = processor.process()

# Define schema and extract structured data
schema = ExtractorSchema(
    fields=[
        Field(name="bill_to", field_type=FieldType.STRING),
        Field(name="invoice_date", field_type=FieldType.DATE),
        Field(name="total_amount", field_type=FieldType.FLOAT)
    ]
)

results = await extractor.extract(documents)
```

#### Image Processing with OCR

```python
config = ProcessingConfig(ocr_enabled=True, ocr_engine="easyocr", language="en")
processor = DocumentProcessor(files=["receipt.jpg"], config=config)

documents = processor.process()
```

---

### 2. Object Detection

IndoxMiner provides powerful **object detection** capabilities with support for a variety of models, such as YOLO, Detectron2, and DETR. Here's how to use these models for real-time image recognition.

#### Supported Models for Object Detection

| Model         | Supported ✅ |
|---------------|:------------:|
| **Detectron2** | ✅          |
| **DETR**       | ✅          |
| **DETR-CLIP**  | ✅          |
| **GroundingDINO** | ✅       |
| **Kosmos2**    | ✅          |
| **OWL-ViT**    | ✅          |
| **RT-DETR**    | ✅          |
| **SAM2**       | ✅          |
| **YOLOv5**     | ✅          |
| **YOLOv6**     | ✅          |
| **YOLOv7**     | ✅          |
| **YOLOv8**     | ✅          |
| **YOLOv10**    | ✅          |
| **YOLOv11**    | ✅          |
| **YOLOX**      | ✅          |
| **YOLO-World**      | ❌          |


---

#### Object Detection with YOLOv5

```python
from indoxminer.detection import YOLOv5

# Initialize YOLOv5 model
detector = YOLOv5()

# Detect objects in an image
image_path = "dog-cat-under-sheet.jpg"
outputs = await detector.detect_objects(image_path)

# Visualize results
detector.visualize_results(outputs)
```

You can also switch to other models by specifying the model name, e.g., `detectron2`, `detr`, `yolov8`, etc.

```python
detector = YOLOv8()  # For YOLOv8
```

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

---

### Configuration Options

#### ProcessingConfig

Customize document processing behavior:

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

---

## 🔍 Error Handling

IndoxMiner provides detailed error reporting for both data extraction and object detection.

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

---

## 🤝 Contributing

We welcome contributions! To contribute:

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

Please read our [Contributing Guidelines](CONTRIBUTING.md) for more details.

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
