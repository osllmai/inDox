---

# Rtf File Loader

## Overview

The `Rtf` function utilizes the `pyth` library to process and extract text from RTF files. It reads the RTF file, converts its content into plain text, and packages this text along with metadata into `Document` objects.

## Installation

Ensure you have the necessary libraries installed:

```bash
pip install pyth indoxArcg
```

## Quick Start

1. **Import the `Rtf` Function**

   ```python
   from indoxArcg.data_loaders import Rtf
   ```

2. **Load and Extract Content from an RTF File**

   - Call the `Rtf` function with the path to your RTF file.

   ```python
   file_path = 'path/to/your/document.rtf'
   documents = Rtf(file_path)
   ```

3. **Access the Document Content and Metadata**

   - The function returns a list of `Document` objects containing the text content and metadata.

   ```python
   for document in documents:
       print(document.metadata)
       print(document.page_content)
   ```

## Function `Rtf`

### Parameters

- `file_path` (str): The path to the RTF file to be loaded.

### Returns

- `List[Document]`: A list containing a `Document` object with the text content of the RTF file and associated metadata.

### Notes

- Metadata includes:
  - `source`: The absolute file path of the RTF file.
  - `page`: A fixed page number set to `1` as RTF files do not inherently have page numbers.
- The text is extracted using `pyth.plugins.rtf15.reader` and converted to plain text using `pyth.plugins.plaintext.writer`.

### Example Usage

```python
from indoxArcg.data_loaders import Rtf

# Load and extract content from the RTF file
file_path = 'document.rtf'
documents = Rtf(file_path)

# Access and print each document's metadata and content
for document in documents:
    print(document.metadata)  # Prints metadata including file path and page number
    print(document.page_content)  # Prints the text content of the RTF file
```

### Error Handling

- Raises `RuntimeError` if there is an issue reading the RTF file or extracting text from it.

---
