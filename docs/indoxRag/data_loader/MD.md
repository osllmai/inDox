---

# Md File Loader

## Overview

The `Md` function loads a Markdown file, extracts its text content, and returns it as a `Document` object with associated metadata. This function is useful for processing Markdown documents, making their content easily accessible for further analysis or storage.

## Installation

Ensure you have the necessary libraries installed:

```bash
pip install indoxRag
```

## Quick Start

1. **Import the `Md` Function**

   ```python
   from indoxRag.data_loaders import Md
   ```

2. **Load and Extract Content from a Markdown File**

   - Call the `Md` function with the file path to your Markdown file.

   ```python
   file_path = 'path/to/your/file.md'
   documents = Md(file_path)
   ```

3. **Access the Document Content and Metadata**

   - The function returns a list containing a `Document` object with the Markdown file's content and metadata.

   ```python
   document = documents[0]
   print(document.metadata)
   print(document.page_content)
   ```

## Function `Md`

### Parameters

- `file_path` (str): The path to the Markdown file to be loaded.

### Returns

- `List[Document]`: A list containing a `Document` object with the text content of the Markdown file and metadata.

### Notes

- Metadata includes the absolute file path (`source`) and a fixed page number (`1`).
- The text content is read directly from the Markdown file without any processing.

### Error Handling

- Raises `RuntimeError` if there is an error loading the Markdown file or processing its content.

### Example Usage

```python
from indoxRag.data_loaders import Md

# Load and extract content from the Markdown file
file_path = 'document.md'
documents = Md(file_path)

# Access and print the document's metadata and content
document = documents[0]
print(document.metadata)  # Prints metadata including file path and page number
print(document.page_content)  # Prints the text content of the Markdown file
```

## Notes

- Ensure the Markdown file exists and is accessible at the provided `file_path`.
- The `page` metadata is set to `1` by default and may be adjusted if necessary.

---
