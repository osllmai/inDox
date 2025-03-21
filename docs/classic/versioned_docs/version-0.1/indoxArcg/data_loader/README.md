# Categories Overview

This documentation organizes supported data loaders into logical categories based on file formats and use cases. Each loader handles specific document types within indoxArcg's RAG/CAG pipeline.

---

### 1. [PDF Processors](PDF-Loaders.md)

**For parsing text and metadata from PDF documents**

- `pdfminer`: Text extraction from PDFs with complex layouts
- `pdfplumber`: Advanced PDF text and table extraction
- `pypdf2`: Basic PDF text extraction and manipulation
- `pypdf4`: Enhanced PDF features support

### 2. [Office Document Loaders](Office-Loaders.md)

**Microsoft Office file format handlers**

- `docx`: Word document processing
- `pptx`: PowerPoint presentation extraction
- `openpyxl`: Excel spreadsheet parsing

### 3. [Structured Data Formats](Structured-Data-Loaders.md)

**Tabular and structured data processors**

- `csv`: Comma-separated values parsing
- `json`: JSON document processing
- `sql`: SQL query results handling
- `md`: Markdown document processing

### 4. [Web & Rich Text](Web-Loaders.md)

**HTML/XML and rich text processors**

- `bs4`: BeautifulSoup HTML/XML parsing
- `rtf`: Rich Text Format processing

### 5. [Scientific Data Formats](Scientific-Loaders.md)

**Specialized scientific data handlers**

- `scipy`: Scientific data file support (.mat, .wav)
- `joblib`: Python object serialization files

### 6. [Text Files](Text-Loaders.md)

**Basic text processing and fallback options**

- `txt`: Plain text file processing

---

## Loader Comparison

| Category             | Loaders              | Text Extraction | Metadata | Images | Tables | Installation Complexity |
| -------------------- | -------------------- | --------------- | -------- | ------ | ------ | ----------------------- |
| PDF Processors       | pdfminer, pypdf\*    | ✅              | ✅       | ❌     | ✅     | Medium                  |
| Office Documents     | docx, pptx, openpyxl | ✅              | ✅       | ✅     | ✅     | Low                     |
| Structured Data      | csv, json, sql       | ✅              | ✅       | ❌     | ✅     | Low                     |
| Web & Markup         | bs4, rtf             | ✅              | ✅       | ❌     | ❌     | Medium                  |
| Scientific Data      | scipy, joblib        | ❌              | ✅       | ✅     | ✅     | High                    |
| Text & Miscellaneous | txt                  | ✅              | ❌       | ❌     | ❌     | None                    |

---

## Quick Start Guide

### Installation

```bash
# PDF Processors
pip install pdfminer.six pdfplumber pypdf2 pypdf4

# Office Documents
pip install python-docx python-pptx openpyxl

# Web Formats
pip install beautifulsoup4 striprtf

# Scientific Data
pip install scipy joblib
```

### Basic Usage

```python
from indoxArcg.data_loaders import PDFLoader, DocxLoader

# Load PDF document
pdf_loader = PDFLoader(pdf_processor='pdfplumber')
pdf_content = pdf_loader.load("document.pdf")

# Load Word document
docx_loader = DocxLoader()
docx_content = docx_loader.load("document.docx")
```

---

## Troubleshooting

1. **Missing Text in PDFs**

   - Try different PDF processors: `pdfplumber` handles complex layouts better
   - Check encrypted files: Use `pypdf2` for password-protected PDFs

2. **Office Document Formatting Issues**

   - Ensure Microsoft Office file versions are supported
   - Use `openpyxl` for Excel files with advanced features

3. **HTML Parsing Errors**

   - Use `bs4` with specific parsers (lxml/html5lib)
   - Handle malformed HTML with `BeautifulSoup(features="html.parser")`

4. **Scientific Data Loading**
   - Verify package versions match file formats
   - Use `joblib` for Python-specific serialization

---

## Detailed Guides

1. [PDF Processing Techniques](PDF-Loaders.md)
2. [Office Document Handling](Office-Loaders.md)
3. [Structured Data Parsing](Structured-Data-Loaders.md)
4. [Web Content Extraction](Web-Loaders.md)
5. [Scientific Data Loading](Scientific-Loaders.md)
6. [Text Processing Basics](Text-Loaders.md)
