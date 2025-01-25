---

# Pptx PowerPoint File Loader

## Overview

The `Pptx` function loads a PowerPoint (.pptx) file and extracts its text content from each slide. Each slide's text is stored in a separate `Document` object with associated metadata. This function is useful for analyzing and processing text data from PowerPoint presentations.

## Installation

Ensure you have the necessary libraries installed:

```bash
pip install python-pptx indoxArcg
```

## Quick Start

1. **Import the `Pptx` Function**

   ```python
   from indoxArcg.data_loaders import Pptx
   ```

2. **Load and Extract Content from a PowerPoint File**

   - Call the `Pptx` function with the file path to your PowerPoint file.

   ```python
   file_path = 'path/to/your/presentation.pptx'
   documents = Pptx(file_path)
   ```

3. **Access the Document Content and Metadata**

   - The function returns a list of `Document` objects, each containing the text content of a slide and associated metadata.

   ```python
   for document in documents:
       print(document.metadata)
       print(document.page_content)
   ```

## Function `Pptx`

### Parameters

- `file_path` (str): The path to the PowerPoint file to be loaded.

### Returns

- `List[Document]`: A list of `Document` objects, each containing the text content of a slide and associated metadata.

### Notes

- Metadata includes:
  - `source`: The absolute file path of the PowerPoint file.
  - `num_slides`: The total number of slides in the presentation.
  - `slide_number`: The slide number (1-based) within the presentation.
- Each slide's text is extracted from its shapes and stored in a separate `Document` object.

### Error Handling

- Raises `RuntimeError` if there is an error loading the PowerPoint file.

### Example Usage

```python
from indoxArcg.data_loaders import Pptx

# Load and extract content from the PowerPoint file
file_path = 'presentation.pptx'
documents = Pptx(file_path)

# Access and print each document's metadata and content
for document in documents:
    print(document.metadata)  # Prints metadata including file path and slide number
    print(document.page_content)  # Prints the text content of the slide
```

## Notes

- Ensure the PowerPoint file exists and is accessible at the provided `file_path`.
- Metadata is generated for each slide, including the slide number and total number of slides.

---
