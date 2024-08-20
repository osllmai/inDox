
---

# Text File Loader

## Overview

The `Txt` function reads a text file, extracts its content, and stores it in a `Document` object with relevant metadata.

## Installation

Ensure you have the necessary libraries installed:

```bash
pip install indox
```

## Quick Start

1. **Import the `Txt` Function**

   ```python
   from your_module import Txt
   ```

2. **Load and Extract Content from a Text File**

   - Call the `Txt` function with the path to your text file.

   ```python
   file_path = 'path/to/your/file.txt'
   documents = Txt(file_path)
   ```

3. **Access the Document Content and Metadata**

   - The function returns a list of `Document` objects containing the text content and metadata.

   ```python
   for document in documents:
       print(document.metadata)
       print(document.page_content)
   ```

## Function `Txt`

### Parameters

- `file_path` (str): The path to the text file to be loaded.

### Returns

- `List[Document]`: A list containing a single `Document` object with the text content of the file and associated metadata.

### Notes

- **Metadata**:
  - `source`: The base name of the text file (i.e., the file name without directory path).
  - `page`: A fixed value of `1`, indicating a single document representing the entire file.

- **Content Handling**:
  - The entire text file is read as a single string and included as text content in the `Document` object.

### Example Usage

```python
from your_module import Txt

# Load and extract content from the text file
file_path = 'file.txt'
documents = Txt(file_path)

# Access and print the document's metadata and content
for document in documents:
    print(document.metadata)  # Prints metadata including the file name
    print(document.page_content)  # Prints the entire text file content
```

### Error Handling

- Raises `RuntimeError` if there is an issue loading the text file or processing its contents.

---

