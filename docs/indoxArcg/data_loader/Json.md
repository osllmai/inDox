---

# Json File Loader

## Overview

The `Json` function loads a JSON file and converts its key-value pairs into a list of `Document` objects. Each key-value pair from the JSON data is stored as a separate `Document` object with associated metadata.

## Installation

Ensure you have the necessary libraries installed:

```bash
pip install indoxArcg
```

## Quick Start

1. **Import the `Json` Function**

   ```python
   from indoxArcg.data_loaders import Json
   ```

2. **Load and Parse a JSON File**

   - Call the `Json` function with the file path to your JSON file.

   ```python
   file_path = 'path/to/your/file.json'
   documents = Json(file_path)
   ```

3. **Access the Document Content and Metadata**

   - The function returns a list of `Document` objects, each containing a JSON key-value pair and associated metadata.

   ```python
   for document in documents:
       print(document.metadata)
       print(document.page_content)
   ```

## Function `Json`

### Parameters

- `file_path` (str): The path to the JSON file to be loaded.

### Returns

- `List[Document]`: A list of `Document` objects, each containing a JSON key-value pair and associated metadata.

### Notes

- Metadata includes the absolute file path (`source`) and the number of entries in the JSON data (`num_entries`).
- Each JSON key-value pair is stored as a string in a separate `Document` object.
- The key is also included in the metadata for each `Document` object.

### Error Handling

- Raises `RuntimeError` if there is an error loading the JSON file or processing its content.

### Example Usage

```python
from indoxArcg.data_loaders import Json

# Load and parse the JSON file
file_path = 'data.json'
documents = Json(file_path)

# Access and print each document's metadata and content
for document in documents:
    print(document.metadata)  # Prints metadata including file path and key
    print(document.page_content)  # Prints the JSON key-value pair
```

## Notes

- Ensure the JSON file exists and is accessible at the provided `file_path`.
- The `page_content` in each `Document` object is a string representation of the JSON key-value pair.

---
