
---

# OpenPyXl Excel File Loader

## Overview

The `OpenPyXl` function loads an Excel file and extracts its data from each sheet, converting it into a list of `Document` objects. Each sheet's data is stored in a separate `Document` object with associated metadata. This function is useful for processing and analyzing data stored in Excel files.

## Installation

Ensure you have the necessary libraries installed:

```bash
pip install pandas openpyxl indox
```

## Quick Start

1. **Import the `OpenPyXl` Function**

   ```python
   from your_module import OpenPyXl
   ```

2. **Load and Extract Content from an Excel File**

   - Call the `OpenPyXl` function with the file path to your Excel file.

   ```python
   file_path = 'path/to/your/file.xlsx'
   documents = OpenPyXl(file_path)
   ```

3. **Access the Document Content and Metadata**

   - The function returns a list of `Document` objects, each containing data from an Excel sheet and associated metadata.

   ```python
   for document in documents:
       print(document.metadata)
       print(document.page_content)
   ```

## Function `OpenPyXl`

### Parameters

- `file_path` (str): The path to the Excel file to be loaded.

### Returns

- `List[Document]`: A list of `Document` objects, each containing the text content of an Excel sheet and associated metadata.

### Notes

- Metadata includes:
  - `source`: The absolute file path of the Excel file.
  - `page`: Set to `1` for all documents (not specific to individual sheets).
- Each sheet in the Excel file is loaded into a separate `Document` object.
- The data in each sheet is converted to a string using `pandas.DataFrame.to_string()` for storage in the `Document` object.
- The `sheet_name` is included in the metadata of each `Document`.

### Error Handling

- Raises `RuntimeError` if there is an error loading the Excel file or processing its content.

### Example Usage

```python
from your_module import OpenPyXl

# Load and extract content from the Excel file
file_path = 'data.xlsx'
documents = OpenPyXl(file_path)

# Access and print each document's metadata and content
for document in documents:
    print(document.metadata)  # Prints metadata including file path and sheet name
    print(document.page_content)  # Prints the text content of the sheet
```

## Notes

- Ensure the Excel file exists and is accessible at the provided `file_path`.
- The metadata `page` is set to `1` for all documents, and you might want to modify this if different handling for pages is needed.

---

