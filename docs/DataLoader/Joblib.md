
---

# Joblib PKL Loader

## Overview

The `Joblib` function loads a Pickle (`.pkl`) or Joblib file, extracts its content, and returns it as a `Document` object with associated metadata. This function is useful for processing serialized data files where the content is stored in a Pickle or Joblib format.

## Installation

Ensure you have the necessary libraries installed:

```bash
pip install joblib indox
```

## Quick Start

1. **Import the `Joblib` Function**

   ```python
   from your_module import Joblib
   ```

2. **Load and Extract Content from a Pickle or Joblib File**

   - Call the `Joblib` function with the file path to your Pickle or Joblib file.

   ```python
   file_path = 'path/to/your/file.pkl'
   documents = Joblib(file_path)
   ```

3. **Access the Document Content and Metadata**

   - The function returns a list containing a `Document` object with the unpickled content and metadata.

   ```python
   document = documents[0]
   print(document.metadata)
   print(document.page_content)
   ```

## Function `Joblib`

### Parameters

- `file_path` (str): The path to the Pickle or Joblib file to be loaded.

### Returns

- `List[Document]`: A list containing a `Document` object with the unpickled content and metadata.

### Notes

- Metadata includes the absolute file path and a default page number of `1`.
- The content is converted to a string before being stored in the `Document` object.

### Error Handling

- Raises `RuntimeError` if there is an error loading the file or processing its content.

### Example Usage

```python
from your_module import Joblib

# Load and extract content from the Pickle or Joblib file
file_path = 'data.pkl'
documents = Joblib(file_path)

# Access the document's metadata and content
document = documents[0]
print(document.metadata)  # Prints metadata like file path and page number
print(document.page_content)  # Prints the unpickled content of the file
```

## Notes

- Ensure the Pickle or Joblib file exists and is accessible at the provided `file_path`.
- The content is converted to a string representation, which may affect data types depending on the original content.

---

