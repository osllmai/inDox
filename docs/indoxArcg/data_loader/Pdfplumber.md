---

# PdfPlumber PDF File Loader

## Overview

The `PdfPlumber` function loads a PDF file and extracts its text content from each page using the `pdfplumber` library. Each page's text is stored in a separate `Document` object with associated metadata. This function is useful for handling and analyzing text data from PDF documents.

## Installation

Ensure you have the necessary libraries installed:

```bash
pip install pdfplumber indoxArcg
```

## Quick Start

1. **Import the `PdfPlumber` Function**

   ```python
   from indoxArcg.data_loaders import PdfPlumber
   ```

2. **Load and Extract Content from a PDF File**

   - Call the `PdfPlumber` function with the file path to your PDF file.

   ```python
   file_path = 'path/to/your/file.pdf'
   documents = PdfPlumber(file_path)
   ```

3. **Access the Document Content and Metadata**

   - The function returns a list of `Document` objects, each containing the text content of a PDF page and associated metadata.

   ```python
   for document in documents:
       print(document.metadata)
       print(document.page_content)
   ```

## Function `PdfPlumber`

### Parameters

- `file_path` (str): The path to the PDF file to be loaded.

### Returns

- `List[Document]`: A list of `Document` objects, each containing the text content of a PDF page and associated metadata.

### Notes

- Metadata includes:
  - `source`: The absolute file path of the PDF file.
  - `page`: The page number of the PDF.
- Each page in the PDF is loaded into a separate `Document` object.
- The text content is extracted using `pdfplumber`'s `extract_text` function.

### Error Handling

- Raises `FileNotFoundError` if the specified file does not exist.
- Raises `RuntimeError` if there is an error reading the PDF file or extracting text.

### Example Usage

```python
from indoxArcg.data_loaders import PdfPlumber

# Load and extract content from the PDF file
file_path = 'document.pdf'
documents = PdfPlumber(file_path)

# Access and print each document's metadata and content
for document in documents:
    print(document.metadata)  # Prints metadata including file path and page number
    print(document.page_content)  # Prints the text content of the page
```

## Notes

- Ensure the PDF file exists and is accessible at the provided `file_path`.
- The `page` metadata is set according to the page number in the PDF.

---
