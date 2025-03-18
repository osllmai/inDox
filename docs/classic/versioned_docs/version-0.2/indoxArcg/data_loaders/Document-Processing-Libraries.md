# Document Processing Libraries

This guide covers advanced document processing frameworks integrated with indoxArcg, providing comprehensive solutions for extracting, parsing, and chunking documents across multiple formats.

---

## Supported Loaders

### Docling

**Best for**: Comprehensive document parsing with intelligent chunking capabilities

#### Features

- Document conversion across multiple formats
- Hybrid chunking with semantic awareness
- Tokenizer integration for optimal text segmentation
- Metadata preservation during processing

```python
from indoxArcg.data_loaders import DoclingReader

# Initialize the reader with a file path
reader = DoclingReader(file_path="document.pdf")

# Load the document
document = reader.load()

# Load and split the document with custom parameters
chunks = reader.load_and_split(max_tokens=512, tokenizer="BAAI/bge-small-en-v1.5")
```

---

### Unstructured

**Best for**: Multi-format document extraction with advanced processing capabilities

#### Features

- Support for diverse file formats (PDF, XLSX, HTML, LaTeX, etc.)
- High-resolution PDF processing
- Title-based chunking
- Metadata extraction and filtering
- Stopword removal option

```python
from indoxArcg.data_loaders import Unstructured

# Initialize with a file path
loader = Unstructured(file_path="document.pdf")

# Load the document
elements = loader.load()

# Load and split with custom parameters
documents = loader.load_and_split(
    remove_stopwords=False,
    max_chunk_size=500
)
```

---

## Key Capabilities

| Feature               | Docling | Unstructured |
| --------------------- | ------- | ------------ |
| PDF Processing        | ✅      | ✅           |
| Office Documents      | ✅      | ✅           |
| Web Content           | ❌      | ✅           |
| LaTeX Support         | ❌      | ✅           |
| Semantic Chunking     | ✅      | ✅           |
| Metadata Preservation | ✅      | ✅           |
| Custom Tokenizers     | ✅      | ❌           |
| Stopword Removal      | ❌      | ✅           |

---

## Installation

```bash
# Install Docling
pip install docling

# Install Unstructured with all dependencies
pip install "unstructured[all]"
```

---

## Advanced Usage

### Docling with Custom Tokenizer

```python
from indoxArcg.data_loaders import DoclingReader

# Initialize with a file path
reader = DoclingReader(file_path="research_paper.pdf")

# Load with page limits
document = reader.load(max_num_pages=10, max_file_size=10_000_000)

# Split with custom tokenizer
chunks = reader.load_and_split(
    max_tokens=1024,
    tokenizer="sentence-transformers/all-MiniLM-L6-v2"
)
```

### Unstructured with Custom Splitter

```python
from indoxArcg.data_loaders import Unstructured
from indoxArcg.splitters import RecursiveCharacterTextSplitter

# Create a custom splitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ". ", " ", ""]
)

# Initialize loader
loader = Unstructured(file_path="complex_document.pdf")

# Load and split with custom splitter
documents = loader.load_and_split(
    max_chunk_size=1000,
    splitter=splitter
)
```

---

## Troubleshooting

### Common Issues

1. **Missing Dependencies**

   ```bash
   # For PDF processing with Unstructured
   pip install "unstructured[pdf]"

   # For Docling PDF support
   pip install "docling[pdf]"
   ```

2. **Memory Issues with Large Documents**

   ```python
   # For Docling, limit pages
   reader = DoclingReader(file_path="large_document.pdf")
   document = reader.load(max_num_pages=50)

   # For Unstructured, use smaller chunk sizes
   loader = Unstructured(file_path="large_document.pdf")
   documents = loader.load_and_split(max_chunk_size=300)
   ```

3. **Tokenizer Not Found**

   ```bash
   # Install transformers and required models
   pip install transformers
   ```

4. **PDF Processing Errors**

   ```python
   # For Unstructured, try different strategies
   from unstructured.partition.pdf import partition_pdf

   elements = partition_pdf(
       filename="problematic.pdf",
       strategy="fast",  # Try "fast" instead of "hi_res"
   )
   ```

---
