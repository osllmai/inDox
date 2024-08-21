
---

# HTML Loader

## Overview

The `Bs4` function loads an HTML file, extracts its text content, and returns it as a list of `Document` objects with minimal metadata. This function is useful for processing HTML documents, making the text content easily accessible for further analysis or storage.

## Installation

Ensure you have the necessary libraries installed:

```bash
pip install beautifulsoup4 indox
```

## Quick Start

1. **Import the `Bs4` Function**

   ```python
   from your_module import Bs4
   ```

2. **Load and Parse an HTML File**

   - Call the `Bs4` function with the file path to your HTML file.

   ```python
   file_path = 'path/to/your/file.html'
   documents = Bs4(file_path)
   ```

3. **Access the Document Content and Metadata**

   - The function returns a list of `Document` objects, each containing the text content and metadata of the HTML file.

   ```python
   document = documents[0]
   print(document.metadata)
   print(document.page_content)
   ```

## Function `Bs4`

### Parameters

- `file_path` (str): The path to the HTML file to be loaded.

### Returns

- `List[Document]`: A list of `Document` objects, each containing the text content of the HTML file and minimal metadata.

### Notes

- Metadata includes the absolute file path and a default page number of `1`.
- The text content is extracted using BeautifulSoup, which handles HTML parsing and text extraction.

### Error Handling

- Raises `FileNotFoundError` if the specified file does not exist.
- Raises `UnicodeDecodeError` if there is an error decoding the HTML file.
- Raises `RuntimeError` for any other errors encountered during HTML processing or document creation.

### Example Usage

```python
from your_module import Bs4

# Load and parse the HTML file
file_path = 'elements.html'
documents = Bs4(file_path)

# Access the document's metadata and content
document = documents[0]
print(document.metadata)  # Prints metadata like file path and page number
print(document.page_content)  # Prints the text content of the HTML file
```

## Notes

- Ensure the HTML file exists and is accessible at the provided `file_path`.
- The `page` metadata is set to `1` by default and may need adjustment based on your specific use case.

---

