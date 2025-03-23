# PDF Loaders

This guide covers PDF processors supported in indoxArcg, organized by capability and use case.

| Platform |
|----------|
| [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/osllmai/inDox/blob/master/cookbook/indoxArcg/PdfLoader.ipynb) |
| [![GitHub](https://img.shields.io/badge/GitHub-Repository-blue?logo=github)](https://github.com/osllmai/inDox/blob/master/cookbook/indoxArcg/PdfLoader.ipynb) |


---

## Table of Contents

- [PDF Loaders](#pdf-loaders)
  - [Table of Contents](#table-of-contents)
  - [Supported Loaders](#supported-loaders)
    - [1. PdfMiner](#1-pdfminer)
      - [Features](#features)
    - [2. PdfPlumber](#2-pdfplumber)
      - [Features](#features-1)
    - [3. PyPDF2](#3-pypdf2)
      - [Features](#features-2)
    - [4. PyPDF4](#4-pypdf4)
      - [Features](#features-3)
  - [Comparison Table](#comparison-table)
  - [Installation](#installation)
  - [Basic Usage](#basic-usage)
    - [Simple Text Extraction](#simple-text-extraction)
    - [Encrypted PDF Handling](#encrypted-pdf-handling)
    - [Table Extraction](#table-extraction)
  - [Advanced Configuration](#advanced-configuration)
    - [Custom Metadata Handling](#custom-metadata-handling)
    - [Parallel Processing](#parallel-processing)
  - [Troubleshooting](#troubleshooting)
    - [Common Issues](#common-issues)
  - [Further Reading](#further-reading)

## Supported Loaders

### 1. PdfMiner

**Best for**: Complex layouts with precise text positioning

#### Features

- Accurate text extraction from multi-column layouts
- Handles PDFs with non-standard encoding
- Page-level metadata tracking

```python
from indoxArcg.data_loaders import PdfMiner

loader = PdfMiner()
docs = loader.load("research_paper.pdf")
```

---

### 2. PdfPlumber

**Best for**: Table extraction and visual debugging

#### Features

- Table data extraction with spatial recognition
- Visual debugging with page images
- Text bounding box analysis

```python
from indoxArcg.data_loaders import PdfPlumber

loader = PdfPlumber(extract_tables=True)
docs = loader.load("financial_report.pdf")
```

---

### 3. PyPDF2

**Best for**: Basic extraction and encryption handling

#### Features

- Password-protected PDF support
- Fast text extraction
- Metadata preservation

```python
from indoxArcg.data_loaders import PyPDF2

loader = PyPDF2(password="your_password")
docs = loader.load("encrypted.pdf")
```

---

### 4. PyPDF4

**Best for**: Advanced PDF features and standards compliance

#### Features

- PDF/A standard support
- Embedded media handling
- Advanced metadata extraction

```python
from indoxArcg.data_loaders import PyPDF4

loader = PyPDF4()
docs = loader.load("specification.pdf")
```

---

## Comparison Table

| Feature             | PdfMiner | PdfPlumber | PyPDF2 | PyPDF4 |
| ------------------- | -------- | ---------- | ------ | ------ |
| Text Accuracy       | ★★★★☆    | ★★★★★      | ★★★☆☆  | ★★★★☆  |
| Table Extraction    | ❌       | ✅         | ❌     | ✅     |
| Password Support    | ❌       | ❌         | ✅     | ✅     |
| Layout Preservation | ✅       | ✅         | ❌     | ❌     |
| Speed               | Medium   | Slow       | Fast   | Medium |
| PDF/A Compliance    | ❌       | ❌         | ❌     | ✅     |

---

## Installation

```bash
pip install pdfminer.six pdfplumber pypdf2 pypdf4
```

---

## Basic Usage

### Simple Text Extraction

```python
from indoxArcg.data_loaders import PdfPlumber

loader = PdfPlumber()
documents = loader.load("document.pdf")
```

### Encrypted PDF Handling

```python
from indoxArcg.data_loaders import PyPDF4

loader = PyPDF4(password="your_password")
documents = loader.load("secured_doc.pdf")
```

### Table Extraction

```python
from indoxArcg.data_loaders import PdfPlumber

loader = PdfPlumber(
    extract_tables=True,
    table_output_format="markdown"  # or "pandas"
)
documents = loader.load("data_report.pdf")
```

---

## Advanced Configuration

### Custom Metadata Handling

```python
from indoxArcg.data_loaders import PdfMiner

def custom_metadata(page_num, text):
    return {"page": page_num + 1, "chars": len(text)}

loader = PdfMiner(metadata_fn=custom_metadata)
```

### Parallel Processing

```python
from indoxArcg.data_loaders import PyPDF4

loader = PyPDF4(thread_count=4)  # Use 4 CPU cores
documents = loader.load("large_document.pdf")
```

---

## Troubleshooting

### Common Issues

1. **Missing Text Content**

   - Try PdfPlumber for complex layouts
   - Enable layout preservation: `PdfMiner(laparams={"all_texts": True})`

2. **Encrypted PDF Errors**

   ```python
   # For PyPDF2/PyPDF4
   loader = PyPDF4(password="correct_password")
   ```

3. **Corrupted Files**

   ```python
   try:
       docs = loader.load("file.pdf")
   except PDFSyntaxError:
       print("Invalid PDF structure")
   ```

4. **Table Extraction Failures**
   - Verify installation: `pip install pdfplumber[csv]`
   - Adjust table detection thresholds:
     ```python
     PdfPlumber(table_settings={"snap_tolerance": 4})
     ```

---

## Further Reading

- [PDF Processing Best Practices](#)
- [Advanced Table Extraction Guide](#)
- [PDF Security Features](#)


Reviewed by: Ali Nemati - March, 22, 2025

*Note: some issue had been reported!*
*lack of demo*