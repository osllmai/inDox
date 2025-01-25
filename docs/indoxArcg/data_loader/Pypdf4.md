---

# PyPdf4 PDF Loader

## Overview

The `PyPdf4` function uses the `PyPDF4` library to read and extract text from a PDF file. It processes each page of the PDF, constructs a `Document` object for each page, and includes relevant metadata. This function is useful for text extraction and processing from PDF documents.

## Installation

Ensure you have the necessary libraries installed:

```bash
pip install PyPDF4 indoxArcg
```

## Quick Start

1. **Import the `PyPdf4` Function**

   ```python
   from indoxArcg.data_loaders import PyPdf4
   ```

2. **Load and Extract Content from a PDF File**

   - Call the `PyPdf4` function with the path to your PDF file.

   ```python
   file_path = 'path/to/your/document.pdf'
   documents = PyPdf4(file_path)
   ```

3. **Access the Document Content and Metadata**

   - The function returns a list of `Document` objects, each containing the text content of a page and associated metadata.

   ```python
   for document in documents:
       print(document.metadata)
       print(document.page_content)
   ```

## Function `PyPdf4`

### Parameters

- `file_path` (str): The path to the PDF file to be loaded.

### Returns

- `List[Document]`: A list of `Document` objects, each containing the text content of a page and associated metadata.

### Notes

- Metadata includes:
  - `source`: The absolute file path of the PDF file.
  - `page`: The page number (0-based index) within the PDF file.
- Text is extracted using `PyPDF4.PdfFileReader`, and each page's text is stored in a separate `Document` object.

### Error Handling

- Raises `FileNotFoundError` if the file does not exist at the specified path.
- Raises `RuntimeError` if there is an issue reading the PDF file or extracting text from any page.

### Example Usage

```python
from indoxArcg.data_loaders import PyPdf4

# Load and extract content from the PDF file
file_path = 'document.pdf'
documents = PyPdf4(file_path)

# Access and print each document's metadata and content
for document in documents:
    print(document.metadata)  # Prints metadata including file path and page number
    print(document.page_content)  # Prints the text content of the page
```

## Notes

- Ensure that the PDF file exists and is accessible at the provided `file_path`.
- The `PyPdf4` function handles the extraction of text from each page of the PDF and collects metadata related to the page and source file.

---
