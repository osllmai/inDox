---

# Csv Loader

## Overview

The `Csv` function is designed to load and parse a CSV file, converting its rows into a list of `Document` objects. This function is particularly useful for situations where CSV data needs to be ingested and analyzed, while maintaining associated metadata for each row.

## Installation

Ensure you have the necessary libraries installed:

```bash
pip install indox
```

## Quick Start

1. **Import the `Csv` Function**

   ```python
   from indoxArcg.data_loaders import CSV
   ```

2. **Load and Parse a CSV File**

   - Call the `CSV` function with the file path to your CSV file.

   ```python
   file_path = 'path/to/your/csv_file.csv'
   documents = CSV(file_path)
   ```

3. **Access the Document Content and Metadata**

   - The function returns a list of `Document` objects, each containing the row content and metadata of the CSV file.

   ```python
   document = documents[0]
   print(document.metadata)
   print(document.page_content)
   ```

## Function `CSV`

### Parameters

- `file_path` (str): The path to the CSV file to be loaded.
- `metadata` (Dict[str, Any], optional): Additional metadata to include in each `Document`. Default is `None`.

### Returns

- `List[Document]`: A list of `Document` objects, each containing the CSV rows and associated metadata.

### Notes

- Metadata can be customized via the `metadata` parameter and will be included in each `Document` object along with the CSV row content.
- The metadata dictionary is initialized with the absolute file path and a default page number of `1`.

### Error Handling

- Raises `FileNotFoundError` if the specified file does not exist.
- Raises `UnicodeDecodeError` if there is an error decoding the CSV file.
- Raises `RuntimeError` for any other errors encountered during CSV processing or document creation.

### Example Usage

```python
from indoxArcg.data_loaders import Csv

# Load and parse the CSV file
file_path = 'data.csv'
documents = CSV(file_path)

# Access the document's metadata and content
document = documents[0]
print(document.metadata)  # Prints metadata like file path and custom key
print(document.page_content)  # Prints the content of the CSV row
```

## Notes

- Ensure the CSV file exists and is accessible at the provided `file_path`.
- The metadata includes the absolute file path and a default page number of `1`.

---
