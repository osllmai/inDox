---

# PyPdf2 PDF Loader

## Overview

The `PyPdf2` function loads a PDF file and extracts text content from each page using the `PyPDF2` library. Each page's text is stored in a separate `Document` object along with relevant metadata. This function is useful for extracting and processing text data from PDF documents.

## Installation

Ensure you have the necessary libraries installed:

```bash
pip install PyPDF2 indoxRag
```

## Quick Start

1. **Import the `PyPdf2` Function**

   ```python
   from indoxRag.data_loaders import PyPdf2
   ```

2. **Load and Extract Content from a PDF File**

   - Call the `PyPdf2` function with the path to your PDF file.

   ```python
   file_path = 'path/to/your/document.pdf'
   documents = PyPdf2(file_path)
   ```

3. **Access the Document Content and Metadata**

   - The function returns a list of `Document` objects, each containing the text content of a page and associated metadata.

   ```python
   for document in documents:
       print(document.metadata)
       print(document.page_content)
   ```

## Function `PyPdf2`

### Parameters

- `file_path` (str): The path to the PDF file to be loaded.

### Returns

- `List[Document]`: A list of `Document` objects, each containing the text content of a page and associated metadata.

### Notes

- Metadata includes:
  - `source`: The absolute file path of the PDF file.
  - `page`: The page number (0-based index) within the PDF file.
- Text is extracted using `PyPDF2.PdfReader`, and each page's text is stored in a separate `Document` object.

### Error Handling

- Raises `FileNotFoundError` if the file does not exist at the specified path.
- Raises `RuntimeError` if there is an issue reading the PDF file or extracting text from any page.

### Example Usage

```python
from indoxRag.data_loaders import PyPdf2

# Load and extract content from the PDF file
file_path = 'document.pdf'
documents = PyPdf2(file_path)

# Access and print each document's metadata and content
for document in documents:
    print(document.metadata)  # Prints metadata including file path and page number
    print(document.page_content)  # Prints the text content of the page
```

## Notes

- Ensure that the PDF file exists and is accessible at the provided `file_path`.
- The `PyPdf2` function handles the extraction of text from each page of the PDF and collects metadata related to the page and source file.

---
