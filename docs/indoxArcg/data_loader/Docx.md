---

# Docx Loader

## Overview

The `Docx` function is designed to load and parse a DOCX file, extracting its text content and estimating page numbers based on paragraph counts. Each page's content is represented as a `Document` object with associated metadata.

## Installation

Ensure you have the necessary libraries installed:

```bash
pip install python-docx indox
```

## Quick Start

1. **Import the `Docx` Function**

   ```python
   from indoxArcg.data_loaders import Docx
   ```

2. **Load and Parse a DOCX File**

   - Call the `Docx` function with the file path to your DOCX file.

   ```python
   file_path = 'path/to/your/document.docx'
   documents = Docx(file_path)
   ```

3. **Access the Document Content and Metadata**

   - The function returns a list of `Document` objects, each containing the text content of a page and metadata.

   ```python
   document = documents[0]
   print(document.metadata)
   print(document.page_content)
   ```

## Function `Docx`

### Parameters

- `file_path` (str): The path to the DOCX file to be loaded.

### Returns

- `List[Document]`: A list of `Document` objects, each containing text content and associated metadata, including estimated page numbers.

### Notes

- The number of pages is estimated based on a fixed number of paragraphs per page (`20` in this case). Adjust `paragraphs_per_page` if a different estimate is required.
- Metadata includes the absolute file path and the estimated page number for each `Document`.

### Error Handling

- Raises `RuntimeError` if there is an error in loading or processing the DOCX file.

### Example Usage

```python
from indoxArcg.data_loaders import Docx

# Load and parse the DOCX file
file_path = 'document.docx'
documents = Docx(file_path)

# Access the document's metadata and content
document = documents[0]
print(document.metadata)  # Prints metadata like file path and page number
print(document.page_content)  # Prints the text content of the page
```

## Notes

- Ensure the DOCX file exists and is accessible at the provided `file_path`.
- The estimated page numbers are based on paragraph counts and may not reflect actual pagination.

---
