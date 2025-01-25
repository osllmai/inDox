---

# Scipy File Loader

## Overview

The `Scipy` function reads a MATLAB `.mat` file, extracts its data, and stores it in `Document` objects. The MATLAB-specific metadata is excluded, and the data variables are included as text content.

## Installation

Ensure you have the necessary libraries installed:

```bash
pip install scipy indoxArcg
```

## Quick Start

1. **Import the `Scipy` Function**

   ```python
   from indoxArcg.data_loaders import Scipy
   ```

2. **Load and Extract Content from a `.mat` File**

   - Call the `Scipy` function with the path to your `.mat` file.

   ```python
   file_path = 'path/to/your/document.mat'
   documents = Scipy(file_path)
   ```

3. **Access the Document Content and Metadata**

   - The function returns a list of `Document` objects containing the data variables and metadata.

   ```python
   for document in documents:
       print(document.metadata)
       print(document.page_content)
   ```

## Function `Scipy`

### Parameters

- `file_path` (str): The path to the `.mat` file to be loaded.

### Returns

- `List[Document]`: A list of `Document` objects, each containing the data variables from the `.mat` file as text content and associated metadata.

### Notes

- **Metadata**:
  - `source`: The absolute file path of the `.mat` file.
  - `page`: An index corresponding to each variable in the `.mat` file, starting from `0`.

- **MATLAB-Specific Metadata**:
  - The `__header__`, `__version__`, and `__globals__` fields are removed from the data before processing.
  
- **Data Handling**:
  - Data variables are converted to strings and included as text content in `Document` objects.

### Example Usage

```python
from indoxArcg.data_loaders import Scipy

# Load and extract content from the .mat file
file_path = 'document.mat'
documents = Scipy(file_path)

# Access and print each document's metadata and content
for document in documents:
    print(document.metadata)  # Prints metadata including file path and variable index
    print(document.page_content)  # Prints the content of the variable from the .mat file
```

### Error Handling

- Raises `RuntimeError` if there is an issue loading the `.mat` file or processing its contents.

---
