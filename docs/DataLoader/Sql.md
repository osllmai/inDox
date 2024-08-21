
---

# SQL File Loader

## Overview

The `Sql` function reads an SQL file, extracts its text content, and stores it in `Document` objects. Metadata includes basic file information and content statistics.

## Installation

Ensure you have the necessary libraries installed:

```bash
pip install indox
```

## Quick Start

1. **Import the `Sql` Function**

   ```python
   from your_module import Sql
   ```

2. **Load and Extract Content from an SQL File**

   - Call the `Sql` function with the path to your SQL file.

   ```python
   file_path = 'path/to/your/file.sql'
   documents = Sql(file_path)
   ```

3. **Access the Document Content and Metadata**

   - The function returns a list of `Document` objects containing the SQL content and metadata.

   ```python
   for document in documents:
       print(document.metadata)
       print(document.page_content)
   ```

## Function `Sql`

### Parameters

- `file_path` (str): The path to the SQL file to be loaded.

### Returns

- `List[Document]`: A list of `Document` objects, each containing the text content of the SQL file and associated metadata.

### Notes

- **Metadata**:
  - `source`: The base name of the SQL file (i.e., the file name without directory path).
  - `page`: A fixed value of `1`, indicating a single document representing the entire file.

- **Content Handling**:
  - The entire SQL file is read as a single string and included as text content in the `Document` object.

### Example Usage

```python
from your_module import Sql

# Load and extract content from the SQL file
file_path = 'file.sql'
documents = Sql(file_path)

# Access and print each document's metadata and content
for document in documents:
    print(document.metadata)  # Prints metadata including the file name
    print(document.page_content)  # Prints the entire SQL file content
```

### Error Handling

- Raises `RuntimeError` if there is an issue loading the SQL file or processing its contents.

---

