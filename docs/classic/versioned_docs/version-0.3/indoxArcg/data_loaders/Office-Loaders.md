# Office Documents

This guide covers Microsoft Office file processors supported in indoxArcg, organized by file type and capability.

---

## Supported Loaders

### 1. Docx Loader

**Best for**: Word document text extraction with paragraph analysis

#### Features

- Paragraph-based page estimation
- Style-aware text extraction
- Metadata preservation

```python
from indoxArcg.data_loaders import Docx

loader = Docx(paragraphs_per_page=25)  # Customize page estimation
docs = loader.load("report.docx")
```

---

### 2. Pptx Loader

**Best for**: PowerPoint presentation content extraction

#### Features

- Slide-level text extraction
- Speaker notes inclusion
- Slide metadata tracking

```python
from indoxArcg.data_loaders import Pptx

loader = Pptx(include_notes=True)
docs = loader.load("presentation.pptx")
```

---

### 3. OpenPyXl Loader

**Best for**: Excel spreadsheet data processing

#### Features

- Multi-sheet handling
- Pandas DataFrame integration
- Cell formatting awareness

```python
from indoxArcg.data_loaders import OpenPyXl

loader = OpenPyXl(sheet_names=["Sales", "Inventory"])
docs = loader.load("data.xlsx")
```

---

## Comparison Table

| Feature             | Docx           | Pptx          | OpenPyXl      |
| ------------------- | -------------- | ------------- | ------------- |
| Text Extraction     | ★★★★★          | ★★★★☆         | ★★★☆☆         |
| Metadata Handling   | Page estimates | Slide numbers | Sheet names   |
| Format Preservation | Paragraphs     | Slide order   | Table formats |
| Image Extraction    | ❌             | ✅ (Beta)     | ❌            |
| Chart Data          | ❌             | ❌            | ✅            |
| Version Support     | 2007+          | 2007+         | .xls/.xlsx    |

---

## Installation

```bash
pip install python-docx python-pptx openpyxl pandas
```

---

## Basic Usage

### Word Document Processing

```python
from indoxArcg.data_loaders import Docx

docs = Docx().load("contract.docx")
```

### PowerPoint Analysis

```python
from indoxArcg.data_loaders import Pptx

loader = Pptx(include_hidden=True)  # Process hidden slides
presentation_docs = loader.load("deck.pptx")
```

### Excel Data Handling

```python
from indoxArcg.data_loaders import OpenPyXl

loader = OpenPyXl(max_rows=1000)  # Limit row processing
spreadsheet_docs = loader.load("dataset.xlsx")
```

---

## Advanced Configuration

### Custom Metadata Handling

```python
def word_metadata(paragraphs, page_num):
    return {"author": "AI Team", "version": 1.2}

loader = Docx(metadata_fn=word_metadata)
```

### Excel Data Processing

```python
from indoxArcg.data_loaders import OpenPyXl

loader = OpenPyXl(
    skip_empty=True,
    data_format="json"  # Alternative: "csv" or "markdown"
)
```

### PowerPoint Image Extraction

```python
from indoxArcg.data_loaders import Pptx

loader = Pptx(
    extract_images=True,
    image_dir="./presentation_images"
)
```

---

## Troubleshooting

### Common Issues

1. **Corrupted Files**

   ```python
   try:
       docs = Docx().load("file.docx")
   except CorruptedFileError:
       print("Invalid Office file format")
   ```

2. **Missing Content**

   - For Excel: Enable formatting awareness
     ```python
     OpenPyXl(preserve_formatting=True)
     ```
   - For PowerPoint: Check slide layouts
     ```python
     Pptx(process_master_slides=True)
     ```

3. **Large File Handling**

   ```python
   Docx(chunk_size=500)  # Process in 500-paragraph chunks
   ```

4. **Version Compatibility**
   - Use `compat_mode=True` for legacy formats:
     ```python
     OpenPyXl(compat_mode=True).load("old_data.xls")
     ```

---
