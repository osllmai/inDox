# Unstructured

## Overview

Unstructured is a versatile document processing library integrated with indoxArcg that excels at extracting and processing content from a wide variety of file formats. It provides robust capabilities for document parsing, chunking, and metadata extraction, making it an essential tool for building comprehensive document processing pipelines.

## Key Features

- **Multi-Format Support**: Process PDFs, Office documents, HTML, LaTeX, and more
- **High-Resolution PDF Processing**: Extract text and structure from complex PDF layouts
- **Title-Based Chunking**: Intelligently split documents based on title structure
- **Metadata Extraction**: Preserve and filter document metadata
- **Stopword Removal**: Option to clean text by removing common stopwords

## Installation

```bash
# Basic installation
pip install unstructured

# With PDF support
pip install "unstructured[pdf]"

# Full installation with all dependencies
pip install "unstructured[all]"
```

## Basic Usage

### Loading a Document

```python
from indoxArcg.data_loaders import Unstructured

# Initialize with a file path
loader = Unstructured(file_path="document.pdf")

# Load the document
elements = loader.load()

# Access document elements
for element in elements:
    print(f"Element type: {element.category}")
    print(f"Content: {element.text}")
    print(f"Metadata: {element.metadata}")
```

### Loading and Splitting a Document

```python
# Load and split with default parameters
documents = loader.load_and_split()

# Load and split with custom parameters
documents = loader.load_and_split(
    remove_stopwords=False,
    max_chunk_size=500
)

# Process the documents
for doc in documents:
    print(f"Content: {doc.page_content}")
    print(f"Metadata: {doc.metadata}")
```

## Advanced Configuration

### Custom Splitting

```python
from indoxArcg.splitters import RecursiveCharacterTextSplitter

# Create a custom splitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ". ", " ", ""]
)

# Use the custom splitter
documents = loader.load_and_split(
    max_chunk_size=1000,
    splitter=splitter
)
```

### Stopword Removal

```python
# Enable stopword removal
documents = loader.load_and_split(
    remove_stopwords=True,
    max_chunk_size=500
)
```

## Implementation Details

The Unstructured class in indoxArcg wraps the core functionality of the Unstructured library:

```python
import importlib
from typing import List

from loguru import logger
import sys

from indoxArcg.core import Document
from indoxArcg.data_loaders.utils import convert_latex_to_md
from indoxArcg.vector_stores.utils import filter_complex_metadata

# Set up logging
logger.remove()  # Remove the default logger
logger.add(
    sys.stdout, format="<green>{level}</green>: <level>{message}</level>", level="INFO"
)

logger.add(
    sys.stdout, format="<red>{level}</red>: <level>{message}</level>", level="ERROR"
)


def import_unstructured_partition(content_type):
    # Import appropriate partition function from the `unstructured` library
    module_name = f"unstructured.partition.{content_type}"
    module = importlib.import_module(module_name)
    partition_function_name = f"partition_{content_type}"
    prt = getattr(module, partition_function_name)
    return prt


def create_documents_unstructured(file_path):
    try:
        if file_path.lower().endswith(".pdf"):
            # Partition PDF with a high-resolution strategy
            from unstructured.partition.pdf import partition_pdf

            elements = partition_pdf(
                filename=file_path,
                strategy="hi_res",
                # infer_table_structure=True,
            )
            # Remove "References" and header elements
            reference_title = [
                el
                for el in elements
                if el.text == "References" and el.category == "Title"
            ][0]
            references_id = reference_title.id
            elements = [el for el in elements if el.metadata.parent_id != references_id]
            elements = [el for el in elements if el.category != "Header"]
        elif file_path.lower().endswith(".xlsx"):
            from unstructured.partition.xlsx import partition_xlsx

            elements_ = partition_xlsx(filename=file_path)
            elements = [el for el in elements_ if el.metadata.text_as_html is not None]
        elif file_path.lower().startswith("www") or file_path.lower().startswith(
            "http"
        ):
            from unstructured.partition.html import partition_html

            elements = partition_html(url=file_path)
        else:
            if file_path.lower().endswith(".tex"):
                file_path = convert_latex_to_md(latex_path=file_path)
            content_type = file_path.lower().split(".")[-1]
            if content_type == "txt":
                prt = import_unstructured_partition(content_type="text")
            else:
                prt = import_unstructured_partition(content_type=content_type)
            elements = prt(filename=file_path)
        return elements
    except AttributeError as ae:
        logger.error(f"Attribute error: {ae}")
        return ae
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return e


def get_chunks_unstructured(file_path, chunk_size, remove_sword, splitter):
    """
    Extract chunks from an unstructured document file using an unstructured data processing library.
    """
    from unstructured.chunking.title import chunk_by_title

    try:
        logger.info("Starting processing")

        # Create initial document elements using the unstructured library
        elements = create_documents_unstructured(file_path)

        if splitter:
            logger.info("Using custom splitter")
            text = ""
            for el in elements:
                text += el.text

            documents = splitter(text=text, max_tokens=chunk_size)
        else:
            logger.info("Using title-based chunking")
            # Split elements based on the title and the specified max characters per chunk
            elements = chunk_by_title(elements, max_characters=chunk_size)
            documents = []

            # Convert each element into a `Document` object with relevant metadata
            for element in elements:
                metadata = element.metadata.to_dict()
                del metadata["languages"]  # Remove unnecessary metadata field

                for key, value in metadata.items():
                    if isinstance(value, list):
                        value = str(value)
                    metadata[key] = value

                if remove_sword:
                    from indoxArcg.data_loader_splitter.utils.clean import (
                        remove_stopwords,
                    )

                    element.text = remove_stopwords(element.text)

                documents.append(
                    Document(page_content=element.text.replace("\n", ""), **metadata)
                )

            # Filter and sanitize complex metadata
            documents = filter_complex_metadata(documents=documents)

        logger.info("Completed chunking process")
        return documents

    except Exception as e:
        logger.error(f"Failed at step with error: {e}")
        raise


class Unstructured:
    def __init__(self, file_path: str):
        """
        Initialize the Unstructured class.
        """
        try:
            self.file_path = file_path
        except Exception as e:
            logger.error(f"Error initializing Unstructured: {e}")
            raise

    def load(self):
        elements = create_documents_unstructured(file_path=self.file_path)
        return elements

    def load_and_split(
        self, remove_stopwords: bool = False, max_chunk_size: int = 500, splitter=None
    ) -> (List)["Document"]:
        """
        Split an unstructured document into chunks.
        """
        try:
            logger.info("Getting all documents")
            docs = get_chunks_unstructured(
                self.file_path, max_chunk_size, remove_stopwords, splitter
            )
            logger.info("Successfully obtained all documents")
            return docs
        except Exception as e:
            logger.error(f"Error in get_all_docs: {e}")
            raise
```

## Supported File Formats

Unstructured supports an extensive range of file formats:

- PDF documents (`.pdf`)
- Microsoft Excel (`.xlsx`)
- Microsoft Word (`.docx`)
- HTML files and web pages
- LaTeX documents (`.tex`)
- Text files (`.txt`)
- Markdown files (`.md`)
- And many more

## Special Features

### PDF Processing

Unstructured provides advanced PDF processing capabilities:

```python
from unstructured.partition.pdf import partition_pdf

# High-resolution strategy for complex PDFs
elements = partition_pdf(
    filename="complex.pdf",
    strategy="hi_res",
)

# Fast strategy for simple PDFs
elements = partition_pdf(
    filename="simple.pdf",
    strategy="fast",
)

# With table structure inference
elements = partition_pdf(
    filename="tables.pdf",
    strategy="hi_res",
    infer_table_structure=True,
)
```

### Web Content Extraction

```python
from indoxArcg.data_loaders import Unstructured

# Load content from a URL
loader = Unstructured(file_path="https://example.com/article")
elements = loader.load()
```

### LaTeX Support

```python
# Automatically converts LaTeX to Markdown before processing
loader = Unstructured(file_path="document.tex")
documents = loader.load_and_split()
```

## Troubleshooting

### Common Issues

1. **Missing Dependencies**

   ```bash
   # For PDF support
   pip install "unstructured[pdf]"

   # For Excel support
   pip install "unstructured[xlsx]"

   # For specific language support
   pip install "unstructured[languages]"
   ```

2. **PDF Processing Errors**

   ```python
   # Try different strategies
   from unstructured.partition.pdf import partition_pdf

   elements = partition_pdf(
       filename="problematic.pdf",
       strategy="fast",  # Try "fast" instead of "hi_res"
   )
   ```

3. **Memory Issues with Large Documents**

   ```python
   # Use smaller chunk sizes
   documents = loader.load_and_split(max_chunk_size=300)
   ```

4. **Metadata Errors**

   ```python
   # Use the filter_complex_metadata utility
   from indoxArcg.vector_stores.utils import filter_complex_metadata

   documents = filter_complex_metadata(documents=documents)
   ```

## Use Cases

- **Document Analysis**: Extract and process text from various document formats
- **Web Scraping**: Extract content from web pages
- **Academic Research**: Process research papers and extract structured information
- **Legal Document Processing**: Extract and chunk legal documents for analysis
- **Content Aggregation**: Process content from multiple sources and formats

## Related Resources

- [Document Processing Libraries](Document-Processing-Libraries.md)
- [PDF Loaders](PDF-Loaders.md)
- [Web Loaders](Web-Loaders.md)
