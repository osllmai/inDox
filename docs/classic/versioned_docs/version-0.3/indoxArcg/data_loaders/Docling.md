# Docling

## Overview

Docling is a powerful document processing library integrated with indoxArcg that provides comprehensive document parsing and intelligent chunking capabilities. It excels at converting documents across multiple formats while preserving their semantic structure.

[Docling GitHub Repository](https://github.com/docling-project/docling)    

## Key Features

- **Document Conversion**: Transform documents between various formats while maintaining structure
- **Hybrid Chunking**: Intelligently split documents using semantic awareness
- **Tokenizer Integration**: Leverage different tokenizers for optimal text segmentation
- **Metadata Preservation**: Retain important document metadata throughout processing

## Installation

```python
# Basic installation
pip install docling
```
# With PDF support
```python
pip install "docling[pdf]"
```
# Full installation with all dependencies
```python
pip install "docling[all]"
```

## Basic Usage

### Loading a Document

```python
from indoxArcg.data_loaders import DoclingReader

# Initialize the reader with a file path
reader = DoclingReader(file_path="document.pdf")

# Load the document
document = reader.load()

# Access document content
text = document.document.export_to_text
```

### Loading and Splitting a Document

```python
# Load and split the document with default parameters
chunks = reader.load_and_split()

# Load and split with custom parameters
chunks = reader.load_and_split(
    max_tokens=512,
    tokenizer="BAAI/bge-small-en-v1.5"
)

# Process the chunks
for chunk in chunks:
    print(f"Chunk: {chunk.text}")
    print(f"Metadata: {chunk.metadata}")
```

## Advanced Configuration

### Limiting Document Size

```python
# Limit by number of pages
document = reader.load(max_num_pages=10)

# Limit by file size (in bytes)
document = reader.load(max_file_size=5_000_000)  # 5MB limit
```

### Custom Tokenizers

```python
# Using a different tokenizer model
chunks = reader.load_and_split(
    max_tokens=1024,
    tokenizer="sentence-transformers/all-MiniLM-L6-v2"
)

# Using a smaller model for faster processing
chunks = reader.load_and_split(
    max_tokens=512,
    tokenizer="distilbert-base-uncased"
)
```

## Implementation Details

The DoclingReader class in indoxArcg wraps the core functionality of the Docling library:

```python
from loguru import logger
import sys

# Set up logging
logger.remove()  # Remove the default logger
logger.add(
    sys.stdout, format="<green>{level}</green>: <level>{message}</level>", level="INFO"
)

logger.add(
    sys.stdout, format="<red>{level}</red>: <level>{message}</level>", level="ERROR"
)


class DoclingReader:
    def __init__(self, file_path: str):
        from docling.document_converter import DocumentConverter

        try:
            self.converter = DocumentConverter()
            self.file_path = file_path
        except Exception as e:
            logger.error(f"Error initializing DocumentConverter: {e}")
            raise

    def load(self, max_num_pages=None, max_file_size=None):
        kwargs = {}

        if max_num_pages is not None:
            kwargs["max_num_pages"] = max_num_pages

        if max_file_size is not None:
            kwargs["max_file_size"] = max_file_size

        self.result = self.converter.convert(source=self.file_path, **kwargs)
        return self.result

    def load_and_split(self, max_tokens=512, tokenizer="BAAI/bge-small-en-v1.5"):
        from docling.chunking import HybridChunker

        # Initialize the chunker with the tokenizer and max tokens
        self.chunker = HybridChunker(tokenizer=tokenizer, max_tokens=max_tokens)

        # Chunk the document
        chunk_iter = self.chunker.chunk(self.result.document)

        return chunk_iter
```

## Supported File Formats

Docling supports a wide range of document formats:

- PDF documents (`.pdf`)
- Microsoft Word documents (`.docx`, `.doc`)
- Text files (`.txt`)
- Markdown files (`.md`)
- HTML files (`.html`, `.htm`)
- And more

## Troubleshooting

### Common Issues

1. **Missing Dependencies**

   ```bash
   # For PDF support
   pip install "docling[pdf]"

   # For OCR capabilities
   pip install "docling[ocr]"
   ```

2. **Memory Issues with Large Documents**

   ```python
   # Limit the number of pages processed
   document = reader.load(max_num_pages=50)

   # Limit the file size
   document = reader.load(max_file_size=10_000_000)  # 10MB
   ```

3. **Tokenizer Not Found**

   ```bash
   # Install transformers and required models
   pip install transformers
   ```

4. **Slow Processing**
   ```python
   # Use a smaller, faster tokenizer
   chunks = reader.load_and_split(
       tokenizer="distilbert-base-uncased"
   )
   ```

## Use Cases

- **Research Paper Analysis**: Extract and chunk academic papers for semantic search
- **Legal Document Processing**: Process contracts and legal documents while preserving structure
- **Content Management**: Convert and chunk content for knowledge bases
- **Data Extraction**: Extract structured information from documents

## Related Resources

- [Document Processing Libraries](Document-Processing-Libraries.md)
- [PDF Loaders](PDF-Loaders.md)
- [Office Document Loaders](Office-Loaders.md)
