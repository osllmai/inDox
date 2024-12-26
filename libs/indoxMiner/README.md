# IndoxMiner

[![PyPI version](https://badge.fury.io/py/indoxminer.svg)](https://badge.fury.io/py/indoxminer)
[![License: MIT](https://img.shields.io/badge/License-AGPL-yellow.svg)](https://opensource.org/licenses/AGPL)

IndoxMiner is a powerful Python library that leverages Large Language Models (LLMs) to extract structured information from unstructured data sources including text, PDFs, and images. Using a flexible schema-based approach, it enables precise data extraction, validation, and transformation, making it ideal for automating document processing workflows.

## 🚀 Key Features

- **Multi-Format Support**: Extract data from text, PDFs, images, and scanned documents
- **Schema-Based Extraction**: Define custom schemas to specify exactly what data to extract
- **LLM Integration**: Seamless integration with OpenAI models for intelligent extraction
- **Validation & Type Safety**: Built-in validation rules and type-safe field definitions
- **Flexible Output**: Export to JSON, pandas DataFrames, or custom formats
- **Async Support**: Built for scalability with asynchronous processing capabilities
- **OCR Integration**: Multiple OCR engine options for image-based text extraction
- **High-Resolution Support**: Enhanced processing for high-quality PDFs
- **Error Handling**: Comprehensive error handling and validation reporting

## 📦 Installation

```bash
pip install indoxminer
```

## 🎯 Quick Start

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

##⚙️ Configuration Options

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